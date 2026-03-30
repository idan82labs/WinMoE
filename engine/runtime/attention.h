#pragma once
/*
 * Attention Mechanism — Grouped Query Attention (GQA)
 *
 * Qwen3.5 uses GQA with:
 *   35B:  num_heads=16, num_kv_heads=2, head_dim=128
 *   397B: num_heads=32, num_kv_heads=2, head_dim=256 (actually uses Gated DeltaNet for 75% of layers)
 *
 * For v0: implement standard dot-product attention.
 * Gated DeltaNet is a later optimization.
 *
 * Per layer:
 *   1. Q = W_q @ x   (hidden -> num_heads * head_dim)
 *   2. K = W_k @ x   (hidden -> num_kv_heads * head_dim)
 *   3. V = W_v @ x   (hidden -> num_kv_heads * head_dim)
 *   4. Apply RoPE to Q, K
 *   5. Attention: score = Q @ K^T / sqrt(d), softmax, out = score @ V
 *   6. O = W_o @ out  (num_heads * head_dim -> hidden)
 *
 * KV Cache: store K,V for all previous tokens to avoid recomputation.
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

/* RMSNorm: y = x * w / sqrt(mean(x^2) + eps) — AVX2 vectorized */
static void rmsnorm(float* out, const float* x, const float* w, int dim) {
    int i;
    /* Pass 1: sum of squares (double precision to match llama.cpp) */
    double ss = 0.0;
    for (i = 0; i < dim; i++) ss += (double)x[i] * (double)x[i];
    float scale = 1.0f / sqrtf((float)(ss / dim) + 1e-6f);

    /* Pass 2: scale and multiply by weight */
    __m256 vscale = _mm256_set1_ps(scale);
    for (i = 0; i + 7 < dim; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vw = _mm256_loadu_ps(w + i);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_mul_ps(vx, vscale), vw));
    }
    for (; i < dim; i++) out[i] = x[i] * ss * w[i];
}

/* RoPE: apply rotary position embedding to Q and K */
static void apply_rope(float* q, float* k, int pos, int head_dim, int num_q_heads, int num_kv_heads, float theta) {
    int h, i;
    /* Apply to each Q head */
    for (h = 0; h < num_q_heads; h++) {
        float* qh = q + h * head_dim;
        for (i = 0; i < head_dim; i += 2) {
            float freq = 1.0f / powf(theta, (float)i / head_dim);
            float val = pos * freq;
            float cos_val = cosf(val);
            float sin_val = sinf(val);
            float q0 = qh[i];
            float q1 = qh[i + 1];
            qh[i]     = q0 * cos_val - q1 * sin_val;
            qh[i + 1] = q0 * sin_val + q1 * cos_val;
        }
    }
    /* Apply to each K head */
    for (h = 0; h < num_kv_heads; h++) {
        float* kh = k + h * head_dim;
        for (i = 0; i < head_dim; i += 2) {
            float freq = 1.0f / powf(theta, (float)i / head_dim);
            float val = pos * freq;
            float cos_val = cosf(val);
            float sin_val = sinf(val);
            float k0 = kh[i];
            float k1 = kh[i + 1];
            kh[i]     = k0 * cos_val - k1 * sin_val;
            kh[i + 1] = k0 * sin_val + k1 * cos_val;
        }
    }
}

/* Softmax in-place */
static void softmax(float* x, int n) {
    float max_val = x[0];
    int i;
    for (i = 1; i < n; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (i = 0; i < n; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    for (i = 0; i < n; i++) x[i] /= sum;
}

/*
 * KV Cache — stores key/value tensors for all past tokens
 */
typedef struct {
    float* keys;    /* [max_seq, num_kv_heads, head_dim] */
    float* values;  /* [max_seq, num_kv_heads, head_dim] */
    int max_seq;
    int num_kv_heads;
    int head_dim;
    int len;        /* current sequence length */
} KVCache;

static void kv_cache_init(KVCache* kv, int max_seq, int num_kv_heads, int head_dim) {
    kv->max_seq = max_seq;
    kv->num_kv_heads = num_kv_heads;
    kv->head_dim = head_dim;
    kv->len = 0;
    int total = max_seq * num_kv_heads * head_dim;
    kv->keys = (float*)calloc(total, sizeof(float));
    kv->values = (float*)calloc(total, sizeof(float));
}

static void kv_cache_append(KVCache* kv, const float* k, const float* v) {
    int kv_dim = kv->num_kv_heads * kv->head_dim;
    memcpy(kv->keys + kv->len * kv_dim, k, kv_dim * sizeof(float));
    memcpy(kv->values + kv->len * kv_dim, v, kv_dim * sizeof(float));
    kv->len++;
}

static void kv_cache_free(KVCache* kv) {
    free(kv->keys);
    free(kv->values);
}

/*
 * Single-token GQA attention
 *
 * q: [num_q_heads * head_dim] — current token's queries
 * kv: KV cache with all past keys/values
 * out: [num_q_heads * head_dim] — attention output
 */
static void gqa_attention(
    float* out,
    const float* q,
    KVCache* kv,
    int num_q_heads,
    int num_kv_heads,
    int head_dim
) {
    int seq_len = kv->len;
    int kv_group = num_q_heads / num_kv_heads; /* how many Q heads per KV head */

    /* For each Q head */
    int h;
    for (h = 0; h < num_q_heads; h++) {
        const float* qh = q + h * head_dim;
        int kv_head = h / kv_group;
        float scale = 1.0f / sqrtf((float)head_dim);

        /* Compute attention scores: Q @ K^T */
        float* scores = (float*)malloc(seq_len * sizeof(float));
        int t;
        for (t = 0; t < seq_len; t++) {
            const float* kt = kv->keys + t * num_kv_heads * head_dim + kv_head * head_dim;
            float dot = 0.0f;
            int d;
            for (d = 0; d < head_dim; d++) dot += qh[d] * kt[d];
            scores[t] = dot * scale;
        }

        /* Softmax */
        softmax(scores, seq_len);

        /* Weighted sum of values */
        float* oh = out + h * head_dim;
        memset(oh, 0, head_dim * sizeof(float));
        for (t = 0; t < seq_len; t++) {
            const float* vt = kv->values + t * num_kv_heads * head_dim + kv_head * head_dim;
            int d;
            for (d = 0; d < head_dim; d++) {
                oh[d] += scores[t] * vt[d];
            }
        }

        free(scores);
    }
}
