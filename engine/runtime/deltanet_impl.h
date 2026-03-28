#pragma once
/*
 * Gated DeltaNet Implementation — The Critical Attention Layer
 *
 * Qwen3.5 uses this for 75% of layers (layers 0,1,2, skip 3, 4,5,6, skip 7, ...)
 *
 * State: 16 groups × [128, 128] per layer = ~1 MB per layer
 * Per token: O(1) state update (no KV cache growth!)
 *
 * References:
 *   - NVlabs/GatedDeltaNet (ICLR 2025)
 *   - justinchuby/qwen3.5-gated-deltanet-analysis
 */

#include <math.h>
#include <string.h>
#include <stdlib.h>

/* DeltaNet config for Qwen3.5-397B:
 * QKV tensor [4096, 12288] = Q(2048) + K(2048) + V(8192)
 * 64 value heads × 128 value_dim, 16 key heads × 128 key_dim
 * State: 64 per-value-head matrices of [128 × 128]
 * Each KV group: 4 value heads share 1 key head (64/16=4)
 */
#define DN_NUM_Q_HEADS 64      /* num_value_heads — output & state heads */
#define DN_NUM_KV_GROUPS 16    /* num_key_heads — K groups */
#define DN_HEADS_PER_GROUP 4   /* 64 / 16 */
#define DN_HEAD_DIM 128        /* both key_head_dim and value_head_dim */
#define DN_INNER 8192          /* 64 * 128 — value/output dimension */
#define DN_KEY_DIM 2048        /* 16 * 128 — key/query dimension */
#define DN_QKV_DIM 12288       /* Q(2048) + K(2048) + V(8192) */
#define DN_GATE_DIM 8192       /* Z gate — same as inner */
#define DN_NUM_GATES 64        /* alpha, beta: one per value head */
#define DN_CONV_WIDTH 4

/* Per-layer DeltaNet state */
typedef struct {
    float* S;           /* [num_q_heads, 128, 128] state matrices */
    float* conv_buf;    /* [4, 8192] conv1d sliding window */
    int conv_pos;       /* current position in conv buffer (circular) */
    /* Pre-allocated temp buffers (avoid malloc per forward call) */
    float* tmp_qkv;     /* [DN_QKV_DIM] */
    float* tmp_z;       /* [DN_GATE_DIM] */
    float* tmp_alpha;   /* [DN_NUM_GATES] */
    float* tmp_beta;    /* [DN_NUM_GATES] */
    float* tmp_head_out;/* [DN_INNER] */
    float* tmp_normed;  /* [DN_INNER] */
    float* tmp_gated;   /* [DN_INNER] */
} DeltaNetState;

static void dn_state_init(DeltaNetState* s) {
    /* Per Q-head state (32 matrices), not per group (16). Each head has own decay. */
    s->S = (float*)calloc(DN_NUM_Q_HEADS * DN_HEAD_DIM * DN_HEAD_DIM, sizeof(float));
    s->conv_buf = (float*)calloc(DN_CONV_WIDTH * DN_QKV_DIM, sizeof(float));
    s->conv_pos = 0;
    s->tmp_qkv = (float*)malloc(DN_QKV_DIM * sizeof(float));
    s->tmp_z = (float*)malloc(DN_GATE_DIM * sizeof(float));
    s->tmp_alpha = (float*)malloc(DN_NUM_GATES * sizeof(float));
    s->tmp_beta = (float*)malloc(DN_NUM_GATES * sizeof(float));
    s->tmp_head_out = (float*)calloc(DN_INNER, sizeof(float));
    s->tmp_normed = (float*)malloc(DN_INNER * sizeof(float));
    s->tmp_gated = (float*)malloc(DN_INNER * sizeof(float));
}

static void dn_state_free(DeltaNetState* s) {
    free(s->S);
    free(s->conv_buf);
    free(s->tmp_qkv); free(s->tmp_z);
    free(s->tmp_alpha); free(s->tmp_beta);
    free(s->tmp_head_out); free(s->tmp_normed); free(s->tmp_gated);
}

/* L2 normalize in-place */
static void dn_l2norm(float* x, int n) {
    float ss = 0.0f;
    int i;
    for (i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / (sqrtf(ss) + 1e-12f);
    for (i = 0; i < n; i++) x[i] *= inv;
}

/* RMS norm (no learned weight version) */
static void dn_rmsnorm_simple(float* out, const float* x, int n) {
    float ss = 0.0f;
    int i;
    for (i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / n + 1e-6f);
    for (i = 0; i < n; i++) out[i] = x[i] * inv;
}

/* RMS norm with learned weight */
static void dn_rmsnorm_weighted(float* out, const float* x, const float* w, int n) {
    float ss = 0.0f;
    int i;
    for (i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / n + 1e-6f);
    for (i = 0; i < n; i++) out[i] = x[i] * inv * w[i];
}

/*
 * Full Gated DeltaNet forward pass for one token
 *
 * Inputs:
 *   x: [hidden_dim] normalized hidden state
 *   state: per-layer recurrent state
 *
 * Weight data (loaded from GGUF):
 *   w_qkv: attn_qkv.weight [hidden, 8192]
 *   w_gate: attn_gate.weight [hidden, 4096]
 *   w_alpha: ssm_alpha.weight [hidden, 32]
 *   w_beta: ssm_beta.weight [hidden, 32]
 *   w_out: ssm_out.weight [4096, hidden]
 *   a_log: ssm_a [32]
 *   dt_bias: ssm_dt.bias [32]
 *   head_norm: ssm_norm.weight [128]
 *
 * Output: [hidden_dim]
 */
static void deltanet_forward(
    float* output,
    const float* x,
    DeltaNetState* state,
    /* Projection weights */
    const void* w_qkv, int w_qkv_type,
    const void* w_gate, int w_gate_type,
    const void* w_alpha, int w_alpha_type,
    const void* w_beta, int w_beta_type,
    const void* w_out, int w_out_type,
    /* Conv1d weight [4, 12288] FP32 */
    const float* conv1d_w,
    /* Learned parameters */
    const float* a_log,
    const float* dt_bias,
    const float* head_norm,
    int hidden_dim
) {
    int i, g, h, d, d2;

    /* 1. Projections — use pre-allocated buffers */
    float* qkv = state->tmp_qkv;
    float* z = state->tmp_z;
    float* alpha_raw = state->tmp_alpha;
    float* beta_raw = state->tmp_beta;

    quant_matvec(qkv, w_qkv, x, DN_QKV_DIM, hidden_dim, w_qkv_type);
    quant_matvec(z, w_gate, x, DN_GATE_DIM, hidden_dim, w_gate_type);

    if (w_alpha) quant_matvec(alpha_raw, w_alpha, x, DN_NUM_GATES, hidden_dim, w_alpha_type);
    else memset(alpha_raw, 0, DN_NUM_GATES * sizeof(float));

    if (w_beta) quant_matvec(beta_raw, w_beta, x, DN_NUM_GATES, hidden_dim, w_beta_type);
    else memset(beta_raw, 0, DN_NUM_GATES * sizeof(float));

    /* 1b. Conv1d: depthwise causal convolution on QKV [kernel=4, channels=12288]
     * For autoregressive: sliding window of last 4 QKV vectors.
     * conv_out[ch] = sum_k(conv_weight[k][ch] * qkv_history[(pos-k) % 4][ch])
     * conv_weight is [4, 12288] FP32. */
    if (state->conv_buf && conv1d_w) {
        /* Store current QKV in circular buffer */
        int conv_slot = state->conv_pos % DN_CONV_WIDTH;
        memcpy(state->conv_buf + conv_slot * DN_QKV_DIM, qkv, DN_QKV_DIM * sizeof(float));
        state->conv_pos++;

        /* Apply depthwise conv1d with SiLU activation */
        for (i = 0; i < DN_QKV_DIM; i++) {
            float sum = 0.0f;
            for (int k = 0; k < DN_CONV_WIDTH; k++) {
                int hist_slot = ((state->conv_pos - 1 - k) % DN_CONV_WIDTH + DN_CONV_WIDTH) % DN_CONV_WIDTH;
                sum += conv1d_w[k * DN_QKV_DIM + i] * state->conv_buf[hist_slot * DN_QKV_DIM + i];
            }
            /* SiLU activation: x * sigmoid(x) */
            float sig = 1.0f / (1.0f + expf(-sum));
            qkv[i] = sum * sig;
        }
    }

    /* 2. Split QKV: Q[16×128=2048] + K[16×128=2048] + V[64×128=8192] */
    float* Q = qkv;                          /* [2048] = 16 key heads × 128 */
    float* K = qkv + DN_KEY_DIM;             /* [2048] = 16 key heads × 128 */
    float* V = qkv + DN_KEY_DIM + DN_KEY_DIM; /* [8192] = 64 value heads × 128 */

    /* 3. Compute gate parameters per head */
    float gate_decay[DN_NUM_GATES];
    float beta[DN_NUM_GATES];

    for (h = 0; h < DN_NUM_GATES; h++) {
        /* Decay: g = exp(-exp(A_log) * softplus(alpha + dt_bias)) */
        float a = (a_log && dt_bias) ?
            expf(-expf(a_log[h]) * (logf(1.0f + expf(alpha_raw[h] + dt_bias[h])))) :
            0.99f;
        /* Clamp to (0, 1] */
        if (a > 1.0f) a = 1.0f;
        if (a < 0.0f) a = 0.0f;
        if (_isnan(a)) a = 0.99f; /* NaN guard (fp:fast safe) */
        gate_decay[h] = a;

        /* Update rate */
        beta[h] = 1.0f / (1.0f + expf(-beta_raw[h])); /* sigmoid */
    }

    /* 4. L2 normalize Q per key-head and K per key-head */
    for (g = 0; g < DN_NUM_KV_GROUPS; g++) {
        dn_l2norm(Q + g * DN_HEAD_DIM, DN_HEAD_DIM);
        dn_l2norm(K + g * DN_HEAD_DIM, DN_HEAD_DIM);
    }

    /* 5. Per-value-head recurrence (64 heads, each with own state + decay) */
    /* Q/K are broadcast from 16 key groups (4 value-heads per key group) */
    float* head_output = state->tmp_head_out;
    memset(head_output, 0, DN_INNER * sizeof(float));
    float scale = 1.0f / sqrtf((float)DN_HEAD_DIM);

    /* Note: do NOT use omp parallel here — the matvec projections above
       already use OpenMP. Nested parallelism causes severe thread contention.
       The recurrence itself is fast enough (~30ms) that parallelizing heads
       within it gives diminishing returns vs the matvec cost (~400ms). */
    for (h = 0; h < DN_NUM_Q_HEADS; h++) {
        g = h / DN_HEADS_PER_GROUP;  /* KV group index (16 key groups) */
        float* S = state->S + h * DN_HEAD_DIM * DN_HEAD_DIM; /* per-value-head state [128,128] */
        float* k = K + g * DN_HEAD_DIM;   /* shared K from key group */
        float* v = V + h * DN_HEAD_DIM;   /* per-value-head V */
        float g_decay = gate_decay[h];     /* per-head gate */
        float b = beta[h];                 /* per-head beta */

        /* === FUSED AVX2 state operations (row-sequential access) === */
        /* Pass 1: Decay + Retrieve + Update — all in one sweep over S rows */
        float retrieved[128];
        float delta[128];
        __m256 vdecay = _mm256_set1_ps(g_decay);

        /* Initialize retrieve accumulator */
        __m256 ret_acc[16]; /* 16 × 8 = 128 floats */
        for (d = 0; d < 16; d++) ret_acc[d] = _mm256_setzero_ps();

        /* Row-by-row: decay S, accumulate k[row] * S_row into retrieved */
        for (d = 0; d < DN_HEAD_DIM; d++) {
            float* S_row = S + d * DN_HEAD_DIM;
            __m256 vk = _mm256_set1_ps(k[d]);
            for (d2 = 0; d2 < DN_HEAD_DIM; d2 += 8) {
                __m256 vs = _mm256_loadu_ps(S_row + d2);
                vs = _mm256_mul_ps(vs, vdecay);           /* decay */
                _mm256_storeu_ps(S_row + d2, vs);          /* store decayed */
                ret_acc[d2 / 8] = _mm256_fmadd_ps(vk, vs, ret_acc[d2 / 8]); /* retrieve */
            }
        }
        /* Store retrieved result */
        for (d = 0; d < 16; d++) _mm256_storeu_ps(retrieved + d * 8, ret_acc[d]);

        /* Delta = beta * (v - retrieved) */
        __m256 vbeta = _mm256_set1_ps(b);
        for (d = 0; d < DN_HEAD_DIM; d += 8) {
            __m256 vv = _mm256_loadu_ps(v + d);
            __m256 vr = _mm256_loadu_ps(retrieved + d);
            _mm256_storeu_ps(delta + d, _mm256_mul_ps(vbeta, _mm256_sub_ps(vv, vr)));
        }

        /* Update state: S += outer(k, delta) — row-sequential */
        for (d = 0; d < DN_HEAD_DIM; d++) {
            float* S_row = S + d * DN_HEAD_DIM;
            __m256 vk = _mm256_set1_ps(k[d]);
            for (d2 = 0; d2 < DN_HEAD_DIM; d2 += 8) {
                __m256 vdelta = _mm256_loadu_ps(delta + d2);
                __m256 vs = _mm256_loadu_ps(S_row + d2);
                _mm256_storeu_ps(S_row + d2, _mm256_fmadd_ps(vk, vdelta, vs));
            }
        }

        /* Pass 2: Output read — o = S^T @ q (row-broadcast FMA pattern) */
        float* q_head = Q + g * DN_HEAD_DIM; /* Q is per key group, not per value head */
        float* o = head_output + h * DN_HEAD_DIM;
        __m256 out_acc[16];
        for (d = 0; d < 16; d++) out_acc[d] = _mm256_setzero_ps();

        for (d = 0; d < DN_HEAD_DIM; d++) {
            float* S_row = S + d * DN_HEAD_DIM;
            __m256 vq = _mm256_set1_ps(q_head[d]);
            for (d2 = 0; d2 < DN_HEAD_DIM; d2 += 8) {
                __m256 vs = _mm256_loadu_ps(S_row + d2);
                out_acc[d2 / 8] = _mm256_fmadd_ps(vq, vs, out_acc[d2 / 8]);
            }
        }
        __m256 vscale = _mm256_set1_ps(scale);
        for (d = 0; d < 16; d++) {
            _mm256_storeu_ps(o + d * 8, _mm256_mul_ps(out_acc[d], vscale));
        }
    }

    /* 6. Per-head RMS norm */
    float* normed_output = state->tmp_normed;
    for (h = 0; h < DN_NUM_Q_HEADS; h++) {
        if (head_norm) {
            dn_rmsnorm_weighted(normed_output + h * DN_HEAD_DIM,
                               head_output + h * DN_HEAD_DIM,
                               head_norm, DN_HEAD_DIM);
        } else {
            dn_rmsnorm_simple(normed_output + h * DN_HEAD_DIM,
                             head_output + h * DN_HEAD_DIM, DN_HEAD_DIM);
        }
    }

    /* 7. Output gating: normed * silu(z) */
    float* gated = state->tmp_gated;
    for (i = 0; i < DN_INNER; i++) {
        float zv = z[i];
        /* Clamp to prevent overflow in silu */
        if (zv > 88.0f) zv = 88.0f;
        if (zv < -88.0f) zv = -88.0f;
        gated[i] = normed_output[i] * (zv / (1.0f + expf(-zv)));
    }

    /* 8. Output projection: gated[4096] → output[hidden_dim] */
    quant_matvec(output, w_out, gated, hidden_dim, DN_INNER, w_out_type);

    /* Buffers are pre-allocated in DeltaNetState — no free needed */
}
