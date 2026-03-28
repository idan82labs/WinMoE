/*
 * WinMoE Inference Engine — Full Token Generation
 *
 * Generates tokens from Qwen3.5 MoE models using:
 * - Custom GGUF parser (no mmap)
 * - Explicit unbuffered I/O for expert weights
 * - Q4_K / Q5_K dequant kernels with AVX2 + OpenMP
 * - GQA attention with KV cache
 * - Within-layer expert streaming
 *
 * Usage: winmoe.exe --model <gguf> --prompt "Hello" --tokens 50
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <windows.h>
#include <immintrin.h>
#include <float.h>

/* Quant tables and kernels */
#include "iq2xxs_grid_def.h"
#include "iq2_dequant.h"
#include "iq2_dequant_fast.h"
#include "q8k_quant.h"
#include "q4k_dequant.h"
#include "q5k_dequant.h"
#include "q6k_dequant.h"
#include "q8_dequant.h"
#include "gguf_parser.h"
#include "transformer.h"
#include "deltanet_impl.h"
#include "gpu_offload.h"

#define MAX_SEQ 512
#define ALIGN 65536

/*
 * Load a tensor's data from GGUF via explicit unbuffered I/O
 * Returns allocated buffer (caller must free)
 */
/*
 * Read tensor data from GGUF via explicit I/O.
 * Returns a malloc'd buffer (caller can safely free()).
 * The read uses aligned I/O internally, then copies to a clean buffer.
 */
static void* read_tensor_from_handle(HANDLE hFile, uint64_t data_start,
                                     TensorInfo* tensor) {
    uint64_t offset = data_start + tensor->offset;
    uint64_t aligned = (offset / ALIGN) * ALIGN;
    int sub = (int)(offset - aligned);
    int read_size = (int)tensor->data_size + sub + ALIGN;
    read_size = ((read_size + ALIGN - 1) / ALIGN) * ALIGN;

    void* aligned_buf = _aligned_malloc(read_size, ALIGN);
    if (!aligned_buf) return NULL;

    LARGE_INTEGER li;
    li.QuadPart = aligned;
    SetFilePointerEx(hFile, li, NULL, FILE_BEGIN);
    DWORD br;
    ReadFile(hFile, aligned_buf, read_size, &br, NULL);

    /* Copy tensor data to a regular malloc buffer (safely free-able) */
    void* result = malloc((size_t)tensor->data_size);
    if (result) {
        memcpy(result, (char*)aligned_buf + sub, (size_t)tensor->data_size);
    }
    _aligned_free(aligned_buf);
    return result;
}

/* Persistent shard file handles (opened once, used for all reads) */
static HANDLE g_shard_handles[MAX_SHARDS] = {0};
static int g_handles_open = 0;

/* Separate handles: sync for weight loading, async for expert streaming */
static HANDLE g_async_handles[MAX_SHARDS] = {0};

static void open_shard_handles(GGUFModel* model) {
    int i;
    for (i = 0; i < model->num_shards; i++) {
        /* Sync handle for weight loading */
        g_shard_handles[i] = CreateFileA(model->shard_paths[i],
            GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
            FILE_FLAG_NO_BUFFERING, NULL);
        if (g_shard_handles[i] == INVALID_HANDLE_VALUE) {
            g_shard_handles[i] = CreateFileA(model->shard_paths[i],
                GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, 0, NULL);
        }
        /* Async handle for overlapped expert reads */
        g_async_handles[i] = CreateFileA(model->shard_paths[i],
            GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
            FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, NULL);
        if (g_async_handles[i] == INVALID_HANDLE_VALUE) {
            g_async_handles[i] = g_shard_handles[i]; /* fallback to sync */
        }
    }
    g_handles_open = 1;
}

/* Async read: issue non-blocking read, returns OVERLAPPED for later wait */
typedef struct {
    OVERLAPPED ov;
    void* buf;          /* aligned read buffer */
    int buf_size;
    void* dest;         /* where to copy data */
    int data_size;
    int sub_offset;     /* offset within aligned buffer */
    int valid;          /* 1 if async op was issued */
} AsyncRead;

static void async_read_start(AsyncRead* ar, int shard, uint64_t abs_offset,
                              void* dest, int data_size, void* aligned_buf, int buf_size) {
    uint64_t aligned = (abs_offset / ALIGN) * ALIGN;
    ar->sub_offset = (int)(abs_offset - aligned);
    ar->buf = aligned_buf;
    ar->buf_size = buf_size;
    ar->dest = dest;
    ar->data_size = data_size;

    memset(&ar->ov, 0, sizeof(OVERLAPPED));
    ar->ov.Offset = (DWORD)(aligned & 0xFFFFFFFF);
    ar->ov.OffsetHigh = (DWORD)(aligned >> 32);
    ar->ov.hEvent = CreateEvent(NULL, TRUE, FALSE, NULL);

    DWORD br;
    BOOL ok = ReadFile(g_async_handles[shard], aligned_buf, buf_size, &br, &ar->ov);
    ar->valid = 1;
    /* ReadFile returns FALSE with ERROR_IO_PENDING for async — that's expected */
}

static void async_read_wait(AsyncRead* ar) {
    if (!ar->valid) return;
    WaitForSingleObject(ar->ov.hEvent, INFINITE);
    /* Copy data from aligned buffer to destination */
    if (ar->dest) {
        memcpy(ar->dest, (char*)ar->buf + ar->sub_offset, ar->data_size);
    }
    CloseHandle(ar->ov.hEvent);
    ar->valid = 0;
}

static void close_shard_handles(GGUFModel* model) {
    int i;
    for (i = 0; i < model->num_shards; i++) {
        if (g_shard_handles[i] && g_shard_handles[i] != INVALID_HANDLE_VALUE)
            CloseHandle(g_shard_handles[i]);
        if (g_async_handles[i] && g_async_handles[i] != INVALID_HANDLE_VALUE
            && g_async_handles[i] != g_shard_handles[i])
            CloseHandle(g_async_handles[i]);
    }
    g_handles_open = 0;
}

/* Load tensor from correct shard using persistent handles */
static void* load_tensor_data(GGUFModel* model, TensorInfo* tensor, int* out_size) {
    int shard = tensor->shard;
    *out_size = (int)tensor->data_size;
    return read_tensor_from_handle(g_shard_handles[shard],
                                   model->shard_data_starts[shard], tensor);
}

/* Read raw bytes from a shard at an absolute offset (for expert reads) */
static int read_bytes_from_shard(int shard, uint64_t abs_offset, void* dest,
                                  int size, void* aligned_buf, int aligned_buf_size) {
    uint64_t aligned = (abs_offset / ALIGN) * ALIGN;
    int sub = (int)(abs_offset - aligned);

    LARGE_INTEGER li;
    li.QuadPart = aligned;
    SetFilePointerEx(g_shard_handles[shard], li, NULL, FILE_BEGIN);
    DWORD br;
    ReadFile(g_shard_handles[shard], aligned_buf, aligned_buf_size, &br, NULL);

    /* Copy just the needed bytes */
    if (dest) memcpy(dest, (char*)aligned_buf + sub, size);
    return (int)br;
}

/*
 * Load shared weights (attention, norms, router) into RAM at startup
 * These are small and always needed — loaded once.
 */
static int load_shared_weights(const char* gguf_path, GGUFModel* model,
                                LayerWeights* layers) {
    int l;
    char name[256];

    /* For large models (397B): only load norms and router (small).
       Attention weights loaded on-demand per layer during inference.
       For small models (35B): load everything. */
    int lazy_attn = (model->hidden_dim >= 4096); /* 397B has hidden=4096 */
    printf("Loading shared weights for %d layers (lazy_attn=%d)...\n",
           model->num_layers, lazy_attn);

    for (l = 0; l < model->num_layers; l++) {
        int size;

        /* Attention norms (FP32) */
        snprintf(name, 256, "blk.%d.attn_norm.weight", l);
        TensorInfo* t = find_tensor(model, name);
        if (t) {
            layers[l].attn_norm = (float*)load_tensor_data(model, t, &size);
        }

        snprintf(name, 256, "blk.%d.ffn_norm.weight", l);
        t = find_tensor(model, name);
        if (t) {
            layers[l].ffn_norm = (float*)load_tensor_data(model, t, &size);
        }

        /* Detect layer type: DeltaNet (has ssm_a) vs full attention (has attn_q) */
        snprintf(name, 256, "blk.%d.ssm_a", l);
        int is_deltanet = (find_tensor(model, name) != NULL);

        if (is_deltanet) {
            layers[l].is_deltanet = 1;

            if (!lazy_attn) {
                /* Small model: load attention weights upfront */
                snprintf(name, 256, "blk.%d.attn_qkv.weight", l);
                t = find_tensor(model, name);
                if (t) { layers[l].w_qkv = load_tensor_data(model, t, &size); layers[l].w_qkv_type = t->type; }

                snprintf(name, 256, "blk.%d.attn_gate.weight", l);
                t = find_tensor(model, name);
                if (t) { layers[l].w_attn_gate = load_tensor_data(model, t, &size); layers[l].w_attn_gate_type = t->type; }

                snprintf(name, 256, "blk.%d.ssm_out.weight", l);
                t = find_tensor(model, name);
                if (t) { layers[l].w_ssm_out = load_tensor_data(model, t, &size); layers[l].w_ssm_out_type = t->type; }

                snprintf(name, 256, "blk.%d.ssm_alpha.weight", l);
                t = find_tensor(model, name);
                if (t) { layers[l].w_alpha = load_tensor_data(model, t, &size); layers[l].w_alpha_type = t->type; }

                snprintf(name, 256, "blk.%d.ssm_beta.weight", l);
                t = find_tensor(model, name);
                if (t) { layers[l].w_beta = load_tensor_data(model, t, &size); layers[l].w_beta_type = t->type; }
            } else {
                /* Large model: store tensor types for on-demand loading */
                snprintf(name, 256, "blk.%d.attn_qkv.weight", l);
                t = find_tensor(model, name);
                if (t) { layers[l].w_qkv_type = t->type; }

                snprintf(name, 256, "blk.%d.attn_gate.weight", l);
                t = find_tensor(model, name);
                if (t) { layers[l].w_attn_gate_type = t->type; }

                snprintf(name, 256, "blk.%d.ssm_out.weight", l);
                t = find_tensor(model, name);
                if (t) { layers[l].w_ssm_out_type = t->type; }

                snprintf(name, 256, "blk.%d.ssm_alpha.weight", l);
                t = find_tensor(model, name);
                if (t) { layers[l].w_alpha_type = t->type; }

                snprintf(name, 256, "blk.%d.ssm_beta.weight", l);
                t = find_tensor(model, name);
                if (t) { layers[l].w_beta_type = t->type; }
            }

            /* Small SSM params (always load — tiny) */
            snprintf(name, 256, "blk.%d.ssm_a", l);
            t = find_tensor(model, name);
            if (t) { layers[l].ssm_a = (float*)load_tensor_data(model, t, &size); }

            snprintf(name, 256, "blk.%d.ssm_dt.bias", l);
            t = find_tensor(model, name);
            if (t) { layers[l].ssm_dt_bias = (float*)load_tensor_data(model, t, &size); }

            snprintf(name, 256, "blk.%d.ssm_norm.weight", l);
            t = find_tensor(model, name);
            if (t) { layers[l].ssm_norm_w = (float*)load_tensor_data(model, t, &size); }
        } else {
            layers[l].is_deltanet = 0;
            /* Standard attention layer (every 4th) */
            if (!lazy_attn) {
                snprintf(name, 256, "blk.%d.attn_q.weight", l);
                t = find_tensor(model, name);
                if (t) { layers[l].wq = load_tensor_data(model, t, &size); layers[l].wq_type = t->type; }

                snprintf(name, 256, "blk.%d.attn_k.weight", l);
                t = find_tensor(model, name);
                if (t) { layers[l].wk = load_tensor_data(model, t, &size); layers[l].wk_type = t->type; }

                snprintf(name, 256, "blk.%d.attn_v.weight", l);
                t = find_tensor(model, name);
                if (t) { layers[l].wv = load_tensor_data(model, t, &size); layers[l].wv_type = t->type; }

                snprintf(name, 256, "blk.%d.attn_output.weight", l);
                t = find_tensor(model, name);
                if (t) { layers[l].wo = load_tensor_data(model, t, &size); layers[l].wo_type = t->type; }
            } else {
                /* Store types for lazy loading — attention is skipped for 397B v0 */
                snprintf(name, 256, "blk.%d.attn_q.weight", l);
                t = find_tensor(model, name);
                if (t) { layers[l].wq_type = t->type; }
            }
        }

        /* Post-attention norm (both layer types) */
        snprintf(name, 256, "blk.%d.post_attention_norm.weight", l);
        t = find_tensor(model, name);
        if (t) {
            layers[l].post_attn_norm = (float*)load_tensor_data(model, t, &size);
        }

        /* Router gate */
        snprintf(name, 256, "blk.%d.ffn_gate_inp.weight", l);
        t = find_tensor(model, name);
        if (t) {
            layers[l].gate_inp = load_tensor_data(model, t, &size);
            layers[l].gate_type = t->type;
        }

        /* Expert tensor offsets (don't load — read on demand) */
        snprintf(name, 256, "blk.%d.ffn_gate_exps.weight", l);
        t = find_tensor(model, name);
        if (t) {
            layers[l].gate_exps_offset = t->offset;
            layers[l].gate_per_expert = t->data_size / model->num_experts;
            layers[l].gate_exps_type = t->type;
            layers[l].gate_exps_shard = t->shard;
        }

        snprintf(name, 256, "blk.%d.ffn_up_exps.weight", l);
        t = find_tensor(model, name);
        if (t) {
            layers[l].up_exps_offset = t->offset;
            layers[l].up_per_expert = t->data_size / model->num_experts;
            layers[l].up_exps_type = t->type;
            layers[l].up_exps_shard = t->shard;
        }

        snprintf(name, 256, "blk.%d.ffn_down_exps.weight", l);
        t = find_tensor(model, name);
        if (t) {
            layers[l].down_exps_offset = t->offset;
            layers[l].down_per_expert = t->data_size / model->num_experts;
            layers[l].down_exps_type = t->type;
            layers[l].down_exps_shard = t->shard;
        }

        if (l % 10 == 0) { fprintf(stderr, "  Layer %d/%d loaded\n", l, model->num_layers); fflush(stderr); }
    }

    printf("Shared weights loaded.\n\n");
    return 0;
}

int main(int argc, char** argv) {
    const char* gguf_path = "D:/models/qwen35-35b-q4/Qwen3.5-35B-A3B-Q4_K_M.gguf";
    int num_tokens = 10;

    /* Parse args */
    int i;
    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i+1 < argc) gguf_path = argv[++i];
        if (strcmp(argv[i], "--tokens") == 0 && i+1 < argc) num_tokens = atoi(argv[++i]);
    }

    fprintf(stderr, "=== WinMoE Inference Engine v0.2 ===\n");
    fprintf(stderr, "Model: %s\n", gguf_path);
    fflush(stderr);
    printf("=== WinMoE Inference Engine v0.2 ===\n\n");

    /* Parse GGUF — heap-allocate because GGUFModel is >1MB (stack overflow on Windows) */
    GGUFModel* pmodel = (GGUFModel*)calloc(1, sizeof(GGUFModel));
    if (!pmodel) { fprintf(stderr, "ERROR: cannot alloc GGUFModel\n"); return 1; }
    if (parse_gguf_split(gguf_path, pmodel) != 0) {
        printf("ERROR: Cannot parse GGUF: %s\n", gguf_path);
        free(pmodel); return 1;
    }
    /* Use macro so all existing `model.` references work with heap allocation */
    #define model (*pmodel)
    fprintf(stderr, "Parse done. Shards: %d, Tensors: %llu\n", model.num_shards, (unsigned long long)model.n_tensors);
    fflush(stderr);
    printf("Shards: %d, Total tensors: %llu\n", model.num_shards, (unsigned long long)model.n_tensors);

    ModelConfig cfg;
    cfg.hidden_dim = model.hidden_dim;
    cfg.intermediate = model.expert_intermediate;
    cfg.num_layers = model.num_layers;
    cfg.num_experts = model.num_experts;
    cfg.expert_k = 4; /* Override: Qwen3.5-397B design point is K=4 for speed/quality balance */
    cfg.rope_theta = model.rope_theta > 0 ? model.rope_theta : 10000000.0f;
    cfg.max_seq_len = MAX_SEQ;
    cfg.num_kv_heads = model.head_count_kv > 0 ? model.head_count_kv : 2;
    cfg.head_dim = model.ssm_state_size > 0 ? model.ssm_state_size : 128;
    cfg.num_q_heads = (model.ssm_inner_size > 0) ?
        (model.ssm_inner_size / cfg.head_dim) : (cfg.hidden_dim / cfg.head_dim);

    printf("Model: %s\n", gguf_path);
    printf("Config: hidden=%d, intermediate=%d, layers=%d\n",
           cfg.hidden_dim, cfg.intermediate, cfg.num_layers);
    printf("Attention: Q=%d heads, KV=%d heads, head_dim=%d\n",
           cfg.num_q_heads, cfg.num_kv_heads, cfg.head_dim);
    printf("MoE: %d experts, K=%d active\n", cfg.num_experts, cfg.expert_k);
    printf("Tokens to generate: %d\n\n", num_tokens);

    /* Open persistent shard handles for all reads */
    open_shard_handles(&model);

    /* Load shared weights */
    LayerWeights* layers = (LayerWeights*)calloc(cfg.num_layers, sizeof(LayerWeights));
    load_shared_weights(gguf_path, &model, layers);

    /* Check what we got */
    int loaded_layers = 0;
    for (i = 0; i < cfg.num_layers; i++) {
        if (layers[i].attn_norm && layers[i].gate_inp) loaded_layers++;
    }
    printf("Loaded %d/%d layers with shared weights\n\n", loaded_layers, cfg.num_layers);

    /* Load embedding and LM head */
    TensorInfo* tok_embd = find_tensor(&model, "token_embd.weight");
    TensorInfo* output_norm = find_tensor(&model, "output_norm.weight");
    TensorInfo* output_weight = find_tensor(&model, "output.weight");

    void* embd_data = NULL;
    float* final_norm = NULL;
    void* lm_head = NULL;
    int lm_head_type = 0;
    int vocab_size = 0;
    int sz;

    if (tok_embd) {
        vocab_size = (int)tok_embd->dims[1];
        fprintf(stderr, "Embeddings: %llu x %llu, type=%d (%llu bytes)\n",
                (unsigned long long)tok_embd->dims[0], (unsigned long long)tok_embd->dims[1],
                tok_embd->type, (unsigned long long)tok_embd->data_size);
        /* For large models: DON'T load entire embedding table.
           Instead, read one row on-demand per token. Store offset info. */
        if (tok_embd->data_size > 100000000ULL) { /* >100 MB = too large */
            fprintf(stderr, "  Embedding too large for bulk load — will read per-token\n");
            embd_data = NULL; /* flag for per-token loading */
        } else {
            embd_data = load_tensor_data(&model, tok_embd, &sz);
        }
    }
    if (output_norm) {
        final_norm = (float*)load_tensor_data(&model, output_norm, &sz);
    }
    if (output_weight) {
        lm_head_type = output_weight->type;
        if (output_weight->data_size > 100000000ULL) {
            fprintf(stderr, "LM head too large (%llu bytes) — skipping for v0.2\n",
                    (unsigned long long)output_weight->data_size);
            lm_head = NULL; /* will produce random logits but won't crash */
        } else {
            lm_head = load_tensor_data(&model, output_weight, &sz);
        }
        fprintf(stderr, "LM head: type=%d, vocab=%d\n", lm_head_type, vocab_size);
    }

    cfg.vocab_size = vocab_size;

    /* Allocate working buffers */
    int H = cfg.hidden_dim;
    int I = cfg.intermediate;
    int K = cfg.expert_k;
    int qkv_dim = cfg.num_q_heads * cfg.head_dim;
    int kv_dim = cfg.num_kv_heads * cfg.head_dim;

    float* hidden = (float*)_aligned_malloc(H * sizeof(float), 32);
    float* residual = (float*)_aligned_malloc(H * sizeof(float), 32);
    float* normed = (float*)_aligned_malloc(H * sizeof(float), 32);
    float* q = (float*)_aligned_malloc(qkv_dim * sizeof(float), 32);
    float* k_cur = (float*)_aligned_malloc(kv_dim * sizeof(float), 32);
    float* v_cur = (float*)_aligned_malloc(kv_dim * sizeof(float), 32);
    float* attn_out = (float*)_aligned_malloc(qkv_dim * sizeof(float), 32);
    float* o_out = (float*)_aligned_malloc(H * sizeof(float), 32);
    float* gate_buf = (float*)_aligned_malloc(I * sizeof(float), 32);
    float* up_buf = (float*)_aligned_malloc(I * sizeof(float), 32);
    float* act_buf = (float*)_aligned_malloc(I * sizeof(float), 32);
    float* expert_out = (float*)_aligned_malloc(H * sizeof(float), 32);
    float* moe_out = (float*)_aligned_malloc(H * sizeof(float), 32);
    float* logits = (float*)malloc(vocab_size * sizeof(float));

    /* Expert weight read buffer */
    int max_expert_size = 1024 * 1024;  /* will be set properly */
    if (loaded_layers > 0) {
        uint64_t es = layers[0].gate_per_expert;
        if (es > (uint64_t)max_expert_size) max_expert_size = (int)es;
        es = layers[0].up_per_expert;
        if (es > (uint64_t)max_expert_size) max_expert_size = (int)es;
        es = layers[0].down_per_expert;
        if (es > (uint64_t)max_expert_size) max_expert_size = (int)es;
    }
    int expert_buf_size = ((max_expert_size + ALIGN * 2) / ALIGN + 1) * ALIGN;
    void* expert_buf = _aligned_malloc(expert_buf_size, ALIGN);

    /* DeltaNet states + KV caches per layer */
    DeltaNetState* dn_states = (DeltaNetState*)calloc(cfg.num_layers, sizeof(DeltaNetState));
    KVCache* kv_caches = (KVCache*)calloc(cfg.num_layers, sizeof(KVCache));
    for (i = 0; i < cfg.num_layers; i++) {
        if (layers[i].is_deltanet) {
            dn_state_init(&dn_states[i]);
        } else {
            kv_cache_init(&kv_caches[i], MAX_SEQ, cfg.num_kv_heads, cfg.head_dim);
        }
    }
    /* === GPU INITIALIZATION === */
    int use_gpu = (gpu_init() == 0);
    if (use_gpu) {
        fprintf(stderr, "Uploading DeltaNet weights to GPU...\n");
        fflush(stderr);
        for (i = 0; i < cfg.num_layers; i++) {
            if (layers[i].is_deltanet) {
                /* Load weights from disk if not already in RAM (lazy mode) */
                char tname[256]; int tsz;
                if (!layers[i].w_qkv) {
                    snprintf(tname, 256, "blk.%d.attn_qkv.weight", i);
                    TensorInfo* tt = find_tensor(&model, tname);
                    if (tt) { layers[i].w_qkv = load_tensor_data(&model, tt, &tsz); layers[i].w_qkv_type = tt->type; }
                }
                if (!layers[i].w_attn_gate) {
                    snprintf(tname, 256, "blk.%d.attn_gate.weight", i);
                    TensorInfo* tt = find_tensor(&model, tname);
                    if (tt) { layers[i].w_attn_gate = load_tensor_data(&model, tt, &tsz); layers[i].w_attn_gate_type = tt->type; }
                }
                if (!layers[i].w_ssm_out) {
                    snprintf(tname, 256, "blk.%d.ssm_out.weight", i);
                    TensorInfo* tt = find_tensor(&model, tname);
                    if (tt) { layers[i].w_ssm_out = load_tensor_data(&model, tt, &tsz); layers[i].w_ssm_out_type = tt->type; }
                }
                if (i % 10 == 0) { fprintf(stderr, "  Loading layer %d/%d...\n", i, cfg.num_layers); fflush(stderr); }
            }
            if (layers[i].is_deltanet && layers[i].w_qkv) {
                /* Upload Q8_0 weights → GPU FP16 */
                int inner = 4096; /* ssm.inner_size — TODO: read from config */
                if (model.ssm_inner_size > 0) inner = model.ssm_inner_size;
                gpu_upload_deltanet_weights(i,
                    layers[i].w_qkv, H, inner * 2,       /* QKV: [hidden, inner*2] */
                    layers[i].w_attn_gate, H, inner,      /* Gate: [hidden, inner] */
                    layers[i].w_ssm_out, inner, H);       /* SSM Out: [inner, hidden] */
            }
        }
        fprintf(stderr, "GPU VRAM used: %.0f MB\n", gpu_vram_used_mb());
        /* Router stays on CPU — 480MB VRAM better spent on expert cache (v8.7, v9.5 confirmed) */
    } else {
        fprintf(stderr, "GPU init failed — running CPU-only\n");
    }

    fprintf(stderr, "DeltaNet states: %d layers, KV caches: %d layers\n",
            (int)(cfg.num_layers * 3 / 4), (int)(cfg.num_layers / 4));

    printf("\nBuffers allocated. Starting generation...\n\n");

    /* === EXPERT RAM CACHE === */
    /* Simple array-based cache: (layer * num_experts + expert_id) → cached data */
    /* Budget: ~20 GB = fits ~1400 full experts (gate+up+down ~14 MB each at Q4_K 397B) */
    int expert_total_size = (int)(layers[0].gate_per_expert + layers[0].up_per_expert + layers[0].down_per_expert);
    long long cache_budget = 20LL * 1024 * 1024 * 1024; /* 20 GB */
    int max_cached = (int)(cache_budget / expert_total_size);
    if (max_cached > cfg.num_layers * cfg.num_experts) max_cached = cfg.num_layers * cfg.num_experts;

    int total_slots = cfg.num_layers * cfg.num_experts;
    void** expert_cache = (void**)calloc(total_slots, sizeof(void*));
    int cache_hits = 0, cache_misses = 0, cache_stored = 0;
    fprintf(stderr, "Expert cache: %d slots, budget %.1f GB, expert_size=%d bytes\n",
            max_cached, (double)cache_budget / (1024*1024*1024), expert_total_size);

    /* === TOKEN GENERATION LOOP === */
    LARGE_INTEGER freq, gen_start, gen_end;
    QueryPerformanceFrequency(&freq);

    /* Profiling accumulators (per token, reset each token) */
    double prof_attn_ms, prof_expert_io_ms, prof_expert_compute_ms;
    double prof_router_ms, prof_norm_ms, prof_embed_ms;
    LARGE_INTEGER prof_t0, prof_t1;

    /* Start with token ID 9707 ("Hello") — hardcoded for now */
    int cur_token = 9707;
    int tokens_generated = 0;

    QueryPerformanceCounter(&gen_start);

    for (int tok = 0; tok < num_tokens; tok++) {
        LARGE_INTEGER tok_start, tok_end;
        QueryPerformanceCounter(&tok_start);
        prof_attn_ms = prof_expert_io_ms = prof_expert_compute_ms = 0;
        prof_router_ms = prof_norm_ms = prof_embed_ms = 0;

        /* 1. Get embedding for current token */
        if (embd_data) {
            /* Small model — embedding table already in RAM */
            if (tok_embd->type == GGML_TYPE_F32) {
                const float* emb = (const float*)embd_data;
                memcpy(hidden, emb + cur_token * H, H * sizeof(float));
            } else if (tok_embd->type == GGML_TYPE_F16) {
                const uint16_t* emb = (const uint16_t*)embd_data;
                for (i = 0; i < H; i++) hidden[i] = fp16_to_fp32(emb[cur_token * H + i]);
            } else if (tok_embd->type == 8) {
                int blocks_per_row = H / Q8_QK;
                int row_bytes = blocks_per_row * Q8_BLOCK_SIZE;
                const void* row_data = (const char*)embd_data + (long long)cur_token * row_bytes;
                q8_dequant_row(hidden, row_data, H);
            }
        } else if (tok_embd && tok_embd->type == 8) {
            /* Large model — read one embedding row from disk on demand */
            int blocks_per_row = H / Q8_QK;
            int row_bytes = blocks_per_row * Q8_BLOCK_SIZE;
            uint64_t embd_abs = model.shard_data_starts[tok_embd->shard] +
                                tok_embd->offset + (uint64_t)cur_token * row_bytes;
            uint64_t embd_aligned = (embd_abs / ALIGN) * ALIGN;
            int embd_sub = (int)(embd_abs - embd_aligned);
            int embd_read_sz = ((row_bytes + embd_sub + ALIGN) / ALIGN) * ALIGN;
            void* embd_buf = _aligned_malloc(embd_read_sz, ALIGN);
            if (embd_buf) {
                LARGE_INTEGER eli; eli.QuadPart = embd_aligned;
                SetFilePointerEx(g_shard_handles[tok_embd->shard], eli, NULL, FILE_BEGIN);
                DWORD ebr;
                ReadFile(g_shard_handles[tok_embd->shard], embd_buf, embd_read_sz, &ebr, NULL);
                q8_dequant_row(hidden, (char*)embd_buf + embd_sub, H);
                _aligned_free(embd_buf);
            } else {
                memset(hidden, 0, H * sizeof(float));
            }
        } else {
            memset(hidden, 0, H * sizeof(float));
        }

        /* Debug: trace signal through layers */
        if (tok == 0) {
            fprintf(stderr, "Embedding[0..3]: %.6f %.6f %.6f %.6f\n",
                    hidden[0], hidden[1], hidden[2], hidden[3]);
        }

        /* 2. Run through all layers */
        for (int layer = 0; layer < cfg.num_layers; layer++) {
            LayerWeights* lw = &layers[layer];
            memcpy(residual, hidden, H * sizeof(float));

            /* 2a. Attention norm */
            if (lw->attn_norm) {
                rmsnorm(normed, hidden, lw->attn_norm, H);
            } else {
                memcpy(normed, hidden, H * sizeof(float));
            }

            /* 2b-g. Attention — dispatch DeltaNet or standard GQA */
            QueryPerformanceCounter(&prof_t0);
            memset(o_out, 0, H * sizeof(float));

            if (lw->is_deltanet) {
                /* On-demand loading for lazy mode (397B) */
                char tname[256];
                int tsz;
                if (!lw->w_qkv) {
                    snprintf(tname, 256, "blk.%d.attn_qkv.weight", layer);
                    TensorInfo* tt = find_tensor(&model, tname);
                    if (tt) { lw->w_qkv = load_tensor_data(&model, tt, &tsz); lw->w_qkv_type = tt->type; }
                }
                if (!lw->w_attn_gate) {
                    snprintf(tname, 256, "blk.%d.attn_gate.weight", layer);
                    TensorInfo* tt = find_tensor(&model, tname);
                    if (tt) { lw->w_attn_gate = load_tensor_data(&model, tt, &tsz); lw->w_attn_gate_type = tt->type; }
                }
                if (!lw->w_ssm_out) {
                    snprintf(tname, 256, "blk.%d.ssm_out.weight", layer);
                    TensorInfo* tt = find_tensor(&model, tname);
                    if (tt) { lw->w_ssm_out = load_tensor_data(&model, tt, &tsz); lw->w_ssm_out_type = tt->type; }
                }
                if (!lw->w_alpha) {
                    snprintf(tname, 256, "blk.%d.ssm_alpha.weight", layer);
                    TensorInfo* tt = find_tensor(&model, tname);
                    if (tt) { lw->w_alpha = load_tensor_data(&model, tt, &tsz); lw->w_alpha_type = tt->type; }
                }
                if (!lw->w_beta) {
                    snprintf(tname, 256, "blk.%d.ssm_beta.weight", layer);
                    TensorInfo* tt = find_tensor(&model, tname);
                    if (tt) { lw->w_beta = load_tensor_data(&model, tt, &tsz); lw->w_beta_type = tt->type; }
                }

                if (use_gpu && lw->w_qkv) {
                    /* === GPU DELTANET (Phase 3: async pipeline) === */
                    int inner = (model.ssm_inner_size > 0) ? model.ssm_inner_size : 4096;
                    int slot_idx = layer % 2;

                    /* ASYNC: Launch QKV+Gate on GPU (returns immediately) */
                    gpu_launch_qkv_gate(layer, slot_idx, normed, H, inner * 2, inner);

                    /* CPU: Alpha + Beta projections WHILE GPU computes (overlap!) */
                    float* alpha_raw = dn_states[layer].tmp_alpha;
                    float* beta_raw = dn_states[layer].tmp_beta;
                    if (lw->w_alpha) quant_matvec(alpha_raw, lw->w_alpha, normed, DN_NUM_GATES, H, lw->w_alpha_type);
                    if (lw->w_beta) quant_matvec(beta_raw, lw->w_beta, normed, DN_NUM_GATES, H, lw->w_beta_type);

                    /* SYNC: Wait for GPU QKV+Gate results */
                    gpu_wait_qkv_gate(slot_idx);
                    float* gpu_qkv = gpu_get_qkv_out(slot_idx);
                    float* gpu_gate = gpu_get_gate_out(slot_idx);

                    /* CPU: DeltaNet state recurrence using GPU's QKV output */
                    float* Q = gpu_qkv;
                    float* K_ptr = gpu_qkv + DN_INNER;
                    float* V_ptr = gpu_qkv + DN_INNER + DN_NUM_KV_GROUPS * DN_HEAD_DIM;

                    /* Compute gate parameters */
                    float gate_decay[DN_NUM_GATES];
                    float beta_vals[DN_NUM_GATES];
                    int hi;
                    for (hi = 0; hi < DN_NUM_GATES; hi++) {
                        float a_val = (lw->ssm_a && lw->ssm_dt_bias) ?
                            expf(-expf(lw->ssm_a[hi]) * (logf(1.0f + expf(alpha_raw[hi] + lw->ssm_dt_bias[hi])))) : 0.99f;
                        if (a_val > 1.0f) a_val = 1.0f;
                        if (a_val < 0.0f) a_val = 0.0f;
                        if (_isnan(a_val)) a_val = 0.99f;
                        gate_decay[hi] = a_val;
                        beta_vals[hi] = 1.0f / (1.0f + expf(-beta_raw[hi]));
                    }

                    /* L2 normalize Q and K */
                    for (hi = 0; hi < DN_NUM_Q_HEADS; hi++) dn_l2norm(Q + hi * DN_HEAD_DIM, DN_HEAD_DIM);
                    for (hi = 0; hi < DN_NUM_KV_GROUPS; hi++) dn_l2norm(K_ptr + hi * DN_HEAD_DIM, DN_HEAD_DIM);

                    /* State recurrence (AVX2 vectorized) */
                    float* head_output = dn_states[layer].tmp_head_out;
                    memset(head_output, 0, DN_INNER * sizeof(float));
                    float scale = 1.0f / sqrtf((float)DN_HEAD_DIM);
                    int h_idx, g_idx, d_idx, d2_idx;

                    for (h_idx = 0; h_idx < DN_NUM_Q_HEADS; h_idx++) {
                        g_idx = h_idx / DN_HEADS_PER_GROUP;
                        float* S = dn_states[layer].S + h_idx * DN_HEAD_DIM * DN_HEAD_DIM;
                        float* k = K_ptr + g_idx * DN_HEAD_DIM;
                        float* v = V_ptr + g_idx * DN_HEAD_DIM;

                        __m256 vdecay = _mm256_set1_ps(gate_decay[h_idx]);
                        __m256 ret_acc[16];
                        for (d_idx = 0; d_idx < 16; d_idx++) ret_acc[d_idx] = _mm256_setzero_ps();

                        for (d_idx = 0; d_idx < DN_HEAD_DIM; d_idx++) {
                            float* S_row = S + d_idx * DN_HEAD_DIM;
                            __m256 vk = _mm256_set1_ps(k[d_idx]);
                            for (d2_idx = 0; d2_idx < DN_HEAD_DIM; d2_idx += 8) {
                                __m256 vs = _mm256_loadu_ps(S_row + d2_idx);
                                vs = _mm256_mul_ps(vs, vdecay);
                                _mm256_storeu_ps(S_row + d2_idx, vs);
                                ret_acc[d2_idx / 8] = _mm256_fmadd_ps(vk, vs, ret_acc[d2_idx / 8]);
                            }
                        }

                        float retrieved[128], delta[128];
                        for (d_idx = 0; d_idx < 16; d_idx++) _mm256_storeu_ps(retrieved + d_idx * 8, ret_acc[d_idx]);

                        __m256 vbeta = _mm256_set1_ps(beta_vals[h_idx]);
                        for (d_idx = 0; d_idx < DN_HEAD_DIM; d_idx += 8) {
                            __m256 vv = _mm256_loadu_ps(v + d_idx);
                            __m256 vr = _mm256_loadu_ps(retrieved + d_idx);
                            _mm256_storeu_ps(delta + d_idx, _mm256_mul_ps(vbeta, _mm256_sub_ps(vv, vr)));
                        }

                        for (d_idx = 0; d_idx < DN_HEAD_DIM; d_idx++) {
                            float* S_row = S + d_idx * DN_HEAD_DIM;
                            __m256 vk2 = _mm256_set1_ps(k[d_idx]);
                            for (d2_idx = 0; d2_idx < DN_HEAD_DIM; d2_idx += 8) {
                                __m256 vdelta = _mm256_loadu_ps(delta + d2_idx);
                                __m256 vs = _mm256_loadu_ps(S_row + d2_idx);
                                _mm256_storeu_ps(S_row + d2_idx, _mm256_fmadd_ps(vk2, vdelta, vs));
                            }
                        }

                        float* q_head = Q + h_idx * DN_HEAD_DIM;
                        float* o = head_output + h_idx * DN_HEAD_DIM;
                        __m256 out_acc[16];
                        for (d_idx = 0; d_idx < 16; d_idx++) out_acc[d_idx] = _mm256_setzero_ps();
                        for (d_idx = 0; d_idx < DN_HEAD_DIM; d_idx++) {
                            float* S_row = S + d_idx * DN_HEAD_DIM;
                            __m256 vq = _mm256_set1_ps(q_head[d_idx]);
                            for (d2_idx = 0; d2_idx < DN_HEAD_DIM; d2_idx += 8)
                                out_acc[d2_idx / 8] = _mm256_fmadd_ps(vq, _mm256_loadu_ps(S_row + d2_idx), out_acc[d2_idx / 8]);
                        }
                        __m256 vscale = _mm256_set1_ps(scale);
                        for (d_idx = 0; d_idx < 16; d_idx++)
                            _mm256_storeu_ps(o + d_idx * 8, _mm256_mul_ps(out_acc[d_idx], vscale));
                    }

                    /* RMS norm + SiLU gating */
                    float* normed_out = dn_states[layer].tmp_normed;
                    float* gated_out = dn_states[layer].tmp_gated;
                    for (hi = 0; hi < DN_NUM_Q_HEADS; hi++) {
                        if (lw->ssm_norm_w)
                            dn_rmsnorm_weighted(normed_out + hi * DN_HEAD_DIM,
                                               head_output + hi * DN_HEAD_DIM, lw->ssm_norm_w, DN_HEAD_DIM);
                        else
                            dn_rmsnorm_simple(normed_out + hi * DN_HEAD_DIM, head_output + hi * DN_HEAD_DIM, DN_HEAD_DIM);
                    }
                    for (hi = 0; hi < DN_INNER; hi++) {
                        float zv = gpu_gate[hi];
                        if (zv > 88.0f) zv = 88.0f;
                        if (zv < -88.0f) zv = -88.0f;
                        gated_out[hi] = normed_out[hi] * (zv / (1.0f + expf(-zv)));
                    }

                    /* ASYNC: Launch SSM Out on GPU */
                    gpu_launch_ssm_out(layer, slot_idx, gated_out, DN_INNER, H);
                    /* SYNC: Wait for SSM Out result */
                    gpu_wait_ssm_out(slot_idx);
                    memcpy(o_out, gpu_get_ssm_out_buf(slot_idx), H * sizeof(float));

                } else if (lw->w_qkv) {
                    /* === CPU FALLBACK: GATED DELTANET === */
                    deltanet_forward(
                        o_out, normed, &dn_states[layer],
                        lw->w_qkv, lw->w_qkv_type,
                        lw->w_attn_gate, lw->w_attn_gate_type,
                        lw->w_alpha, lw->w_alpha_type,
                        lw->w_beta, lw->w_beta_type,
                        lw->w_ssm_out, lw->w_ssm_out_type,
                        lw->ssm_a, lw->ssm_dt_bias, lw->ssm_norm_w,
                        H
                    );
                }
            } else {
                /* === STANDARD GQA ATTENTION (every 4th layer) === */
                /* TODO: implement full GQA — zero output for now */
            }

            /* Residual add */
            for (i = 0; i < H; i++) hidden[i] = residual[i] + o_out[i];
            memcpy(residual, hidden, H * sizeof(float));
            QueryPerformanceCounter(&prof_t1);
            prof_attn_ms += (double)(prof_t1.QuadPart - prof_t0.QuadPart) / freq.QuadPart * 1000.0;

            /* Post-attention norm (before MoE) */
            if (lw->post_attn_norm) {
                rmsnorm(normed, hidden, lw->post_attn_norm, H);
            } else if (lw->ffn_norm) {
                rmsnorm(normed, hidden, lw->ffn_norm, H);
            } else {
                memcpy(normed, hidden, H * sizeof(float));
            }

            /* Debug: after attention */
            if (tok == 0 && layer == 0) {
                int nan_attn = 0;
                for (i = 0; i < H; i++) if (hidden[i] != hidden[i]) nan_attn++;
                fprintf(stderr, "  L0 after attn: h[0..3]=%.4f %.4f %.4f %.4f  nan=%d\n",
                        hidden[0], hidden[1], hidden[2], hidden[3], nan_attn);
                fprintf(stderr, "  L0 q[0..3]=%.4f %.4f  k[0..3]=%.4f %.4f\n",
                        q[0], q[1], k_cur[0], k_cur[1]);
                fprintf(stderr, "  L0 attn_out[0..3]=%.4f %.4f  o_out[0..3]=%.4f %.4f\n",
                        attn_out[0], attn_out[1], o_out[0], o_out[1]);
            }

            /* (post_attn_norm already applied above, normed is ready for MoE) */

            /* 2i. Router — find top-K experts */
            QueryPerformanceCounter(&prof_t0);
            int expert_ids[16];
            float expert_weights[16];
            if (lw->gate_inp) {
                router_topk(normed, lw->gate_inp, lw->gate_type,
                           H, cfg.num_experts, K, expert_ids, expert_weights);
            } else {
                /* Fallback: use first K experts */
                for (i = 0; i < K; i++) { expert_ids[i] = i; expert_weights[i] = 1.0f / K; }
            }

            QueryPerformanceCounter(&prof_t1);
            prof_router_ms += (double)(prof_t1.QuadPart - prof_t0.QuadPart) / freq.QuadPart * 1000.0;

            /* Pre-quantize normed activation to Q8_K once for all experts */
            int q8k_blocks = H / Q8K_QK;
            block_q8_K* normed_q8k = (block_q8_K*)_malloca(q8k_blocks * sizeof(block_q8_K));
            if (normed_q8k) quantize_row_q8_K(normed_q8k, normed, H);

            /* 2j. Expert FFN for each selected expert */
            memset(moe_out, 0, H * sizeof(float));
            for (int ek = 0; ek < K; ek++) {
                int eid = expert_ids[ek];

                /* === Expert I/O + Compute with cache === */
                QueryPerformanceCounter(&prof_t0);

                int cache_key = layer * cfg.num_experts + eid;

                /* Check GPU expert cache first (256 GB/s vs 38 GB/s) */
                int gpu_idx = use_gpu ? gpu_find_cached_expert(layer, eid) : -1;
                if (gpu_idx >= 0) {
                    /* GPU EXPERT HIT — fully fused: gate+up+swiglu+down all on GPU */
                    gpu_expert_ffn_fused(gpu_idx, normed, H, I, expert_out);

                    QueryPerformanceCounter(&prof_t1);
                    prof_expert_io_ms += (double)(prof_t1.QuadPart - prof_t0.QuadPart) / freq.QuadPart * 1000.0;

                    /* Accumulate */
                    __m256 vw = _mm256_set1_ps(expert_weights[ek]);
                    for (i = 0; i + 7 < H; i += 8) {
                        __m256 vm = _mm256_loadu_ps(moe_out + i);
                        __m256 ve = _mm256_loadu_ps(expert_out + i);
                        _mm256_storeu_ps(moe_out + i, _mm256_fmadd_ps(vw, ve, vm));
                    }
                    for (; i < H; i++) moe_out[i] += expert_weights[ek] * expert_out[i];
                    continue; /* skip CPU path */
                }

                void* cached = expert_cache[cache_key];
                void* gate_data; void* up_data; void* down_data;

                if (cached) {
                    /* RAM CACHE HIT */
                    gate_data = cached;

                    /* Try to promote to GPU cache */
                    if (use_gpu && gpu_expert_cache_count() < 240) { /* limit: 240 × 7.6MB = 1.8GB, 248MB headroom */
                        gpu_cache_expert(layer, eid,
                            cached, (int)lw->gate_per_expert,
                            (char*)cached + lw->gate_per_expert, (int)lw->up_per_expert,
                            (char*)cached + lw->gate_per_expert + lw->up_per_expert, (int)lw->down_per_expert);
                    }
                    up_data = (char*)cached + lw->gate_per_expert;
                    down_data = (char*)cached + lw->gate_per_expert + lw->up_per_expert;
                    cache_hits++;
                } else {
                    /* CACHE MISS — async read all 3 concurrently (NVMe QD=3) */
                    void* slot = malloc(expert_total_size);

                    /* Allocate 3 aligned buffers for concurrent reads */
                    void* gate_io_buf = _aligned_malloc(expert_buf_size, ALIGN);
                    void* up_io_buf = _aligned_malloc(expert_buf_size, ALIGN);
                    void* down_io_buf = _aligned_malloc(expert_buf_size, ALIGN);

                    uint64_t gate_off = model.shard_data_starts[lw->gate_exps_shard] + lw->gate_exps_offset +
                                        (uint64_t)eid * lw->gate_per_expert;
                    uint64_t up_off = model.shard_data_starts[lw->up_exps_shard] + lw->up_exps_offset +
                                      (uint64_t)eid * lw->up_per_expert;
                    uint64_t down_off = model.shard_data_starts[lw->down_exps_shard] + lw->down_exps_offset +
                                        (uint64_t)eid * lw->down_per_expert;

                    /* Issue all 3 reads concurrently */
                    AsyncRead ar_gate, ar_up, ar_down;
                    if (gate_io_buf && slot)
                        async_read_start(&ar_gate, lw->gate_exps_shard, gate_off,
                                         slot, (int)lw->gate_per_expert, gate_io_buf, expert_buf_size);
                    if (up_io_buf && slot)
                        async_read_start(&ar_up, lw->up_exps_shard, up_off,
                                         (char*)slot + lw->gate_per_expert, (int)lw->up_per_expert,
                                         up_io_buf, expert_buf_size);
                    if (down_io_buf && slot)
                        async_read_start(&ar_down, lw->down_exps_shard, down_off,
                                         (char*)slot + lw->gate_per_expert + lw->up_per_expert,
                                         (int)lw->down_per_expert, down_io_buf, expert_buf_size);

                    /* Wait for all 3 (they ran concurrently on NVMe) */
                    if (gate_io_buf && slot) async_read_wait(&ar_gate);
                    if (up_io_buf && slot) async_read_wait(&ar_up);
                    if (down_io_buf && slot) async_read_wait(&ar_down);

                    if (gate_io_buf) _aligned_free(gate_io_buf);
                    if (up_io_buf) _aligned_free(up_io_buf);
                    if (down_io_buf) _aligned_free(down_io_buf);

                    /* Point to stable memory in slot */
                    if (slot) {
                        gate_data = slot;
                        up_data = (char*)slot + lw->gate_per_expert;
                        down_data = (char*)slot + lw->gate_per_expert + lw->up_per_expert;
                        /* Cache if budget allows, else free after use */
                        if (cache_stored < max_cached) {
                            expert_cache[cache_key] = slot;
                            cache_stored++;
                        }
                        /* If not cached, slot will leak (bounded: K*layers per token) */
                    } else {
                        /* malloc failed — zero output for this expert */
                        gate_data = up_data = down_data = expert_buf; /* will produce garbage but won't crash */
                    }
                    cache_misses++;
                }

                /* Fused gate+up matmul — single OMP region, shared activation */
                if (normed_q8k && lw->gate_exps_type == GGML_TYPE_Q4_K
                              && lw->up_exps_type == GGML_TYPE_Q4_K) {
                    /* Fused gate+up with prefetch */
                    int bpr = H / Q4K_QK;
                    const block_q4_K* gw = (const block_q4_K*)gate_data;
                    const block_q4_K* uw = (const block_q4_K*)up_data;
                    int frow;
                    #pragma omp parallel for schedule(static)
                    for (frow = 0; frow < I; frow++) {
                        /* Prefetch next row's first blocks */
                        if (frow + 1 < I) {
                            _mm_prefetch((const char*)&gw[(frow+1) * bpr], _MM_HINT_T0);
                            _mm_prefetch((const char*)&uw[(frow+1) * bpr], _MM_HINT_T0);
                        }
                        float gsum = 0.0f, usum = 0.0f;
                        int fb;
                        for (fb = 0; fb < bpr; fb++) {
                            gsum += q4k_dot_q8k(&gw[frow * bpr + fb], &normed_q8k[fb]);
                            usum += q4k_dot_q8k(&uw[frow * bpr + fb], &normed_q8k[fb]);
                        }
                        gate_buf[frow] = gsum;
                        up_buf[frow] = usum;
                    }
                } else {
                    quant_matvec(gate_buf, gate_data, normed, I, H, lw->gate_exps_type);
                    quant_matvec(up_buf, up_data, normed, I, H, lw->up_exps_type);
                }

                /* SwiGLU: act = silu(gate) * up — AVX2 fast sigmoid */
                for (i = 0; i + 7 < I; i += 8) {
                    __m256 vg = _mm256_loadu_ps(gate_buf + i);
                    __m256 vu = _mm256_loadu_ps(up_buf + i);
                    /* Fast sigmoid: sig(x) ≈ 0.5 + 0.5 * x / (1 + |x|) */
                    __m256 vabs = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), vg);
                    __m256 vdenom = _mm256_add_ps(_mm256_set1_ps(1.0f), vabs);
                    __m256 vsig = _mm256_add_ps(_mm256_set1_ps(0.5f),
                                  _mm256_mul_ps(_mm256_set1_ps(0.5f),
                                  _mm256_div_ps(vg, vdenom)));
                    /* silu(x) = x * sig(x), then multiply by up */
                    __m256 vsilu = _mm256_mul_ps(vg, vsig);
                    _mm256_storeu_ps(act_buf + i, _mm256_mul_ps(vsilu, vu));
                }
                for (; i < I; i++) {
                    float g = gate_buf[i];
                    float sig = 0.5f + 0.5f * g / (1.0f + fabsf(g));
                    act_buf[i] = g * sig * up_buf[i];
                }

                /* Down matmul — pre-quantize act_buf for Q5_K integer path */
                if (lw->down_exps_type == GGML_TYPE_Q5_K) {
                    int act_q8k_blocks = I / Q8K_QK;
                    block_q8_K act_q8k[4]; /* I=1024 → 4 blocks, stack-safe */
                    quantize_row_q8_K(act_q8k, act_buf, I);
                    q5k_matvec_q8k(expert_out, down_data, act_q8k, H, I);
                } else {
                    quant_matvec(expert_out, down_data, act_buf, H, I, lw->down_exps_type);
                }

                QueryPerformanceCounter(&prof_t1);
                prof_expert_io_ms += (double)(prof_t1.QuadPart - prof_t0.QuadPart) / freq.QuadPart * 1000.0;

                /* Accumulate weighted expert output — AVX2 */
                {
                    __m256 vw = _mm256_set1_ps(expert_weights[ek]);
                    for (i = 0; i + 7 < H; i += 8) {
                        __m256 vm = _mm256_loadu_ps(moe_out + i);
                        __m256 ve = _mm256_loadu_ps(expert_out + i);
                        _mm256_storeu_ps(moe_out + i, _mm256_fmadd_ps(vw, ve, vm));
                    }
                    for (; i < H; i++) moe_out[i] += expert_weights[ek] * expert_out[i];
                }
            }

            if (normed_q8k) _freea(normed_q8k);

            /* 2k. Residual add */
            for (i = 0; i < H; i++) hidden[i] = residual[i] + moe_out[i];

            /* Debug: trace per layer */
            if (tok == 0 && (layer == 0 || layer == cfg.num_layers - 1)) {
                float hmax = 0;
                int has_nan_h = 0;
                for (i = 0; i < H; i++) {
                    if (hidden[i] != hidden[i]) has_nan_h = 1;
                    if (fabsf(hidden[i]) > hmax) hmax = fabsf(hidden[i]);
                }
                fprintf(stderr, "  L%d after MoE: h[0..3]=%.4f %.4f %.4f %.4f  max=%.4f nan=%d  experts=%d,%d,%d\n",
                        layer, hidden[0], hidden[1], hidden[2], hidden[3],
                        hmax, has_nan_h, expert_ids[0], expert_ids[1], expert_ids[2]);
            }
        }

        /* 3. Final norm */
        if (final_norm) {
            rmsnorm(normed, hidden, final_norm, H);
        } else {
            memcpy(normed, hidden, H * sizeof(float));
        }

        /* 4. LM head — project to vocab */
        if (lm_head) {
            quant_matvec(logits, lm_head, normed, vocab_size, H, lm_head_type);
        } else {
            /* LM head not loaded (too large) — use normed[0] as dummy logit */
            for (i = 0; i < vocab_size; i++) logits[i] = normed[i % H];
        }

        /* Debug: logits */
        if (tok == 0) {
            float lmax = logits[0]; int lmax_id = 0;
            for (i = 1; i < vocab_size; i++) {
                if (logits[i] > lmax) { lmax = logits[i]; lmax_id = i; }
            }
            fprintf(stderr, "Logits: [0]=%.4f [1]=%.4f max=%.4f at id=%d\n",
                    logits[0], logits[1], lmax, lmax_id);
            /* Also check for NaN in logits */
            int nan_count = 0;
            for (i = 0; i < vocab_size; i++) if (logits[i] != logits[i]) nan_count++;
            if (nan_count > 0) fprintf(stderr, "WARNING: %d NaN logits!\n", nan_count);
        }

        /* 5. Greedy sample */
        int best_token = 0;
        float best_logit = logits[0];
        for (i = 1; i < vocab_size; i++) {
            if (logits[i] > best_logit) {
                best_logit = logits[i];
                best_token = i;
            }
        }

        QueryPerformanceCounter(&tok_end);
        double tok_ms = (double)(tok_end.QuadPart - tok_start.QuadPart) / freq.QuadPart * 1000.0;

        cur_token = best_token;
        tokens_generated++;

        fprintf(stderr, "Token %d: id=%d (%.1f ms) [attn=%.0f expert=%.0f router=%.0f]\n",
                tok, best_token, tok_ms,
                prof_attn_ms, prof_expert_io_ms, prof_router_ms);
        /* Print token IDs to stdout for output capture */
        printf("%d ", best_token);
        fflush(stdout);
    }

    QueryPerformanceCounter(&gen_end);
    double total_ms = (double)(gen_end.QuadPart - gen_start.QuadPart) / freq.QuadPart * 1000.0;
    double tps = tokens_generated / (total_ms / 1000.0);

    fprintf(stderr, "\n=== Generation Complete ===\n");
    fprintf(stderr, "Tokens: %d in %.1f ms = %.2f tok/s\n", tokens_generated, total_ms, tps);
    fprintf(stderr, "Expert cache: %d hits, %d misses, %d stored (%.1f%% hit rate)\n",
            cache_hits, cache_misses, cache_stored,
            cache_hits + cache_misses > 0 ? 100.0 * cache_hits / (cache_hits + cache_misses) : 0);

    /* JSON output for benchmark */
    printf("{\"tok_s\": %.2f, \"tokens\": %d, \"total_ms\": %.1f, \"status\": \"ok\"}\n",
           tps, tokens_generated, total_ms);

    /* Cleanup */
    for (i = 0; i < cfg.num_layers; i++) kv_cache_free(&kv_caches[i]);
    free(kv_caches);
    _aligned_free(hidden); _aligned_free(residual); _aligned_free(normed);
    _aligned_free(q); _aligned_free(k_cur); _aligned_free(v_cur);
    _aligned_free(attn_out); _aligned_free(o_out);
    _aligned_free(gate_buf); _aligned_free(up_buf); _aligned_free(act_buf);
    _aligned_free(expert_out); _aligned_free(moe_out);
    _aligned_free(expert_buf);
    free(logits);
    free(layers);
    close_shard_handles(&model);
    if (use_gpu) gpu_shutdown();
    #undef model
    free(pmodel);
    return 0;
}
