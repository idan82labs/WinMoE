#pragma once
/*
 * Gated DeltaNet — Linear Attention for Qwen3.5
 *
 * 75% of layers use this instead of standard softmax attention.
 * Key advantage: O(1) memory per token (fixed-size state matrix vs KV cache).
 *
 * Per-token recurrence:
 *   g_t = exp(-exp(A_log) * softplus(alpha + dt_bias))  // decay gate
 *   beta_t = sigmoid(beta_raw)                           // update rate
 *   q_t = l2norm(q), k_t = l2norm(k)                   // normalized Q,K
 *   state = g_t * state + outer(k_t, beta_t * (v_t - state^T @ k_t))
 *   output = state^T @ q_t / sqrt(dk)
 *
 * State: [num_heads, head_dim, head_dim_v] — fixed size, no KV cache growth
 */

#include <math.h>
#include <string.h>
#include <stdlib.h>

/* L2 normalize a vector in-place */
static void l2_normalize(float* x, int dim) {
    float ss = 0.0f;
    int i;
    for (i = 0; i < dim; i++) ss += x[i] * x[i];
    float inv = 1.0f / (sqrtf(ss) + 1e-12f);
    for (i = 0; i < dim; i++) x[i] *= inv;
}

/* Softplus: log(1 + exp(x)) */
static float softplus(float x) {
    if (x > 20.0f) return x;  /* avoid overflow */
    return logf(1.0f + expf(x));
}

/* Sigmoid */
static float sigmoid_f(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/* SiLU (Swish): x * sigmoid(x) */
static float silu_f(float x) {
    return x / (1.0f + expf(-x));
}

/*
 * Gated DeltaNet State — per layer
 */
typedef struct {
    float* state;        /* [num_heads, dk, dv] */
    float* conv_state;   /* [conv_width, qkv_dim] for causal conv1d */
    int num_heads;
    int dk;
    int dv;
    int conv_width;
} GDNState;

static void gdn_state_init(GDNState* s, int num_heads, int dk, int dv, int conv_width) {
    s->num_heads = num_heads;
    s->dk = dk;
    s->dv = dv;
    s->conv_width = conv_width;
    s->state = (float*)calloc(num_heads * dk * dv, sizeof(float));
    s->conv_state = (float*)calloc(conv_width * (num_heads * (dk + dv) + num_heads * dk), sizeof(float));
}

static void gdn_state_free(GDNState* s) {
    free(s->state);
    free(s->conv_state);
}

/*
 * Gated DeltaNet forward pass for one token
 *
 * x: [hidden_dim] input
 * state: per-layer recurrent state
 * weights: projection weights from GGUF
 *
 * Simplified implementation — handles the core recurrence
 */
static void gated_deltanet_forward(
    float* out,          /* [hidden_dim] output */
    const float* x,      /* [hidden_dim] input */
    GDNState* state,
    /* Weight data pointers */
    const void* w_qkvz,  /* fused QKV+Z projection */
    int w_qkvz_type,
    const void* w_ba,    /* beta+alpha projection (gate params) */
    int w_ba_type,
    const void* w_out,   /* output projection */
    int w_out_type,
    /* Learned parameters */
    const float* a_log,  /* [num_heads] decay rate */
    const float* dt_bias, /* [num_heads] time-step bias */
    /* Config */
    int hidden_dim,
    int num_heads,
    int head_dim
) {
    int H = num_heads;
    int dk = head_dim;
    int dv = head_dim;
    int qkvz_dim = H * (dk + dk + dv + dv); /* Q + K + V + Z */
    int ba_dim = H * 2; /* beta + alpha per head */

    /* 1. Project to Q, K, V, Z */
    float* qkvz = (float*)malloc(qkvz_dim * sizeof(float));
    quant_matvec(qkvz, w_qkvz, x, qkvz_dim, hidden_dim, w_qkvz_type);

    /* Split into Q, K, V, Z */
    float* q_raw = qkvz;
    float* k_raw = qkvz + H * dk;
    float* v_raw = qkvz + H * dk + H * dk;
    float* z_raw = qkvz + H * dk + H * dk + H * dv;

    /* 2. Project gate parameters */
    float* ba = (float*)malloc(ba_dim * sizeof(float));
    if (w_ba) {
        quant_matvec(ba, w_ba, x, ba_dim, hidden_dim, w_ba_type);
    } else {
        memset(ba, 0, ba_dim * sizeof(float));
    }

    /* 3. Per-head recurrence */
    float* head_out = (float*)malloc(H * dv * sizeof(float));

    int h;
    for (h = 0; h < H; h++) {
        float* q = q_raw + h * dk;
        float* k = k_raw + h * dk;
        float* v = v_raw + h * dv;
        float* S = state->state + h * dk * dv;  /* state matrix for this head */

        /* Gate parameters */
        float beta_raw = ba[h];
        float alpha_raw = ba[H + h];

        float beta = sigmoid_f(beta_raw);

        /* Decay gate: g = exp(-exp(A_log) * softplus(alpha + dt_bias)) */
        float a = (a_log && dt_bias) ?
                  expf(-expf(a_log[h]) * softplus(alpha_raw + dt_bias[h])) : 0.99f;
        float g = a;  /* decay ∈ (0, 1] */

        /* L2 normalize Q and K */
        l2_normalize(q, dk);
        l2_normalize(k, dk);

        /* Decay old state */
        int ij;
        for (ij = 0; ij < dk * dv; ij++) {
            S[ij] *= g;
        }

        /* Retrieve: what does state predict for this key? */
        /* retrieved = S^T @ k (result is dv-dimensional) */
        float* retrieved = (float*)malloc(dv * sizeof(float));
        int d, d2;
        for (d = 0; d < dv; d++) {
            float sum = 0.0f;
            for (d2 = 0; d2 < dk; d2++) {
                sum += S[d2 * dv + d] * k[d2];
            }
            retrieved[d] = sum;
        }

        /* Delta = beta * (v - retrieved) */
        float* delta = (float*)malloc(dv * sizeof(float));
        for (d = 0; d < dv; d++) {
            delta[d] = beta * (v[d] - retrieved[d]);
        }

        /* Update state: S += outer(k, delta) */
        for (d = 0; d < dk; d++) {
            for (d2 = 0; d2 < dv; d2++) {
                S[d * dv + d2] += k[d] * delta[d2];
            }
        }

        /* Read output: o = S^T @ q / sqrt(dk) */
        float scale = 1.0f / sqrtf((float)dk);
        float* oh = head_out + h * dv;
        for (d = 0; d < dv; d++) {
            float sum = 0.0f;
            for (d2 = 0; d2 < dk; d2++) {
                sum += S[d2 * dv + d] * q[d2];
            }
            oh[d] = sum * scale;
        }

        free(retrieved);
        free(delta);
    }

    /* 4. Output gating: out = norm(head_out) * silu(z) */
    /* Simplified: just multiply by silu(z) per element */
    float* gated = (float*)malloc(H * dv * sizeof(float));
    for (h = 0; h < H; h++) {
        for (int d = 0; d < dv; d++) {
            int idx = h * dv + d;
            gated[idx] = head_out[idx] * silu_f(z_raw[idx]);
        }
    }

    /* 5. Output projection */
    quant_matvec(out, w_out, gated, hidden_dim, H * dv, w_out_type);

    free(qkvz);
    free(ba);
    free(head_out);
    free(gated);
}
