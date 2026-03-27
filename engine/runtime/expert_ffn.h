#pragma once
/*
 * Expert FFN — Gate + Up → SwiGLU → Down
 *
 * For Qwen3.5-397B-A17B:
 *   hidden_dim = 4096
 *   expert_intermediate = 1024
 *
 * Expert FFN(x):
 *   gate = W_gate @ x    (1024 x 4096)
 *   up   = W_up   @ x    (1024 x 4096)
 *   act  = SiLU(gate) * up   (element-wise, 1024)
 *   out  = W_down  @ act  (4096 x 1024)
 *   return out
 *
 * Total: 3 matmuls per expert, K=3 experts per layer = 9 matmuls/layer
 */

#include <math.h>

#define HIDDEN_DIM 4096
#define EXPERT_INTERMEDIATE 1024

/* SiLU activation: x * sigmoid(x) */
static inline float silu(float x) {
    return x / (1.0f + expf(-x));
}

/* SwiGLU: silu(gate) * up, element-wise */
static inline void swiglu(float* out, const float* gate, const float* up, int dim) {
    int i;
    for (i = 0; i < dim; i++) {
        out[i] = silu(gate[i]) * up[i];
    }
}

/*
 * Run one expert FFN: x (4096) → out (4096)
 *
 * expert_data: pointer to expert weight block from slab
 *   Layout: [gate_proj weights][up_proj weights][down_proj weights]
 *   gate: 1024 x 4096 in IQ2_XXS
 *   up:   1024 x 4096 in IQ2_XXS (different size due to quant grouping)
 *   down: 4096 x 1024 in IQ2_XXS
 *
 * Buffers: gate_buf, up_buf, act_buf must be pre-allocated
 */
static inline void expert_ffn(
    float* out,               /* output: [4096] */
    const float* x,           /* input: [4096] */
    const void* expert_data,  /* raw expert weights from slab */
    float* gate_buf,          /* temp: [1024] */
    float* up_buf,            /* temp: [1024] */
    float* act_buf,           /* temp: [1024] */
    int gate_size,            /* bytes of gate_proj weights */
    int up_size               /* bytes of up_proj weights */
) {
    /* Skip 4-byte header per component (tensor type/shape metadata) */
    const void* gate_weights = (const char*)expert_data + 4;
    const void* up_weights = (const char*)expert_data + gate_size + 4;
    const void* down_weights = (const char*)expert_data + gate_size + up_size + 4;

    /* gate = W_gate @ x */
    iq2xxs_matvec_fast(gate_buf, gate_weights, x, EXPERT_INTERMEDIATE, HIDDEN_DIM);

    /* up = W_up @ x */
    iq2xxs_matvec_fast(up_buf, up_weights, x, EXPERT_INTERMEDIATE, HIDDEN_DIM);

    /* act = SwiGLU(gate, up) */
    swiglu(act_buf, gate_buf, up_buf, EXPERT_INTERMEDIATE);

    /* out = W_down @ act */
    iq2xxs_matvec_fast(out, down_weights, act_buf, HIDDEN_DIM, EXPERT_INTERMEDIATE);
}

/*
 * Run MoE layer: route to K experts, compute FFNs, combine
 *
 * expert_outputs: [K][4096] — pre-allocated
 * routing_weights: [K] — from router softmax
 */
static inline void moe_combine(
    float* out,                    /* output: [4096] = residual + weighted sum */
    const float* residual,         /* input: [4096] */
    const float expert_outputs[][HIDDEN_DIM], /* K expert outputs */
    const float* routing_weights,  /* K weights from router */
    int K
) {
    int i, k;
    for (i = 0; i < HIDDEN_DIM; i++) {
        float sum = 0.0f;
        for (k = 0; k < K; k++) {
            sum += routing_weights[k] * expert_outputs[k][i];
        }
        out[i] = residual[i] + sum;
    }
}
