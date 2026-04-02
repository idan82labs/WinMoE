#pragma once
/*
 * Minimal GGUF Parser — extract tensor metadata for expert weight loading
 *
 * We only need: tensor name, type, dimensions, and data offset.
 * No mmap, no full model loading — just read the header to build
 * an offset table for direct I/O reads.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define MAX_TENSORS 4000
#define MAX_SHARDS 8
#define MAX_NAME_LEN 256
#define MAX_DIMS 4

/* GGML type IDs we care about */
#define GGML_TYPE_F32    0
#define GGML_TYPE_F16    1
#define GGML_TYPE_Q4_0   2
#define GGML_TYPE_Q8_0   8
#define GGML_TYPE_Q4_K  12
#define GGML_TYPE_Q5_K  13
#define GGML_TYPE_Q6_K  14
#define GGML_TYPE_IQ2_XXS 16
#define GGML_TYPE_IQ1_M  22

/* Block sizes per type */
static int ggml_block_size(int type) {
    switch (type) {
        case GGML_TYPE_F32:    return 4;    /* 1 weight per 4 bytes */
        case GGML_TYPE_F16:    return 2;
        case GGML_TYPE_Q4_0:   return 18;   /* 32 weights per block */
        /* Q8_0 intentionally NOT here — data_size stays FP32 to avoid read issues */
        case GGML_TYPE_Q4_K:   return 144;  /* 256 weights per block */
        case GGML_TYPE_Q5_K:   return 176;
        case GGML_TYPE_Q6_K:   return 210;
        case GGML_TYPE_IQ2_XXS: return 66;
        default: return 0;
    }
}

static int ggml_block_weights(int type) {
    switch (type) {
        case GGML_TYPE_F32:    return 1;
        case GGML_TYPE_F16:    return 1;
        case GGML_TYPE_Q4_0:   return 32;
        case GGML_TYPE_Q4_K:   return 256;
        case GGML_TYPE_Q5_K:   return 256;
        case GGML_TYPE_Q6_K:   return 256;
        case GGML_TYPE_IQ2_XXS: return 256;
        default: return 1;
    }
}

typedef struct {
    char name[MAX_NAME_LEN];
    int type;
    int n_dims;
    uint64_t dims[MAX_DIMS];
    uint64_t offset;     /* byte offset in GGUF file (relative to data_start) */
    uint64_t data_size;  /* total bytes of tensor data */
    int shard;           /* which shard file this tensor lives in (0-based) */
} TensorInfo;

typedef struct {
    uint32_t version;
    uint64_t n_tensors;
    TensorInfo tensors[MAX_TENSORS];
    uint64_t data_start;  /* byte offset where tensor data begins (per shard) */
    int num_shards;
    char shard_paths[MAX_SHARDS][512];
    uint64_t shard_data_starts[MAX_SHARDS];

    /* Model config (extracted from KV pairs) */
    int hidden_dim;
    int expert_intermediate;
    int feed_forward_length;  /* for shared expert (non-MoE FFN) */
    int num_experts;
    int num_layers;
    int expert_used_count;
    int head_count;
    int head_count_kv;
    int ssm_state_size;
    int ssm_inner_size;
    int ssm_conv_kernel;
    float rope_theta;
    float routed_scaling_factor;  /* MoE expert output scaling (DeepSeek/Qwen3.5 style) */
} GGUFModel;

/* Read little-endian uint64 */
static uint64_t read_u64(FILE* f) {
    uint64_t v; fread(&v, 8, 1, f); return v;
}
static uint32_t read_u32(FILE* f) {
    uint32_t v; fread(&v, 4, 1, f); return v;
}
static uint16_t read_u16(FILE* f) {
    uint16_t v; fread(&v, 2, 1, f); return v;
}

/* Read GGUF string (length-prefixed) */
static int read_gguf_string(FILE* f, char* buf, int max_len) {
    uint64_t len = read_u64(f);
    if (len >= (uint64_t)max_len) {
        _fseeki64(f, (long long)len, SEEK_CUR);
        buf[0] = 0;
        return (int)len;
    }
    fread(buf, 1, (size_t)len, f);
    buf[len] = 0;
    return (int)len;
}

/* Skip a KV value based on type */
static void skip_kv_value(FILE* f, uint32_t vtype) {
    static const int sizes[] = {1,1,2,2,4,4,4,8,0,0,1};
    if (vtype == 8) { /* string */
        uint64_t len = read_u64(f);
        _fseeki64(f, (long long)len, SEEK_CUR);
    } else if (vtype == 9) { /* array */
        uint32_t atype = read_u32(f);
        uint64_t alen = read_u64(f);
        if (atype == 8) {
            uint64_t i;
            for (i = 0; i < alen; i++) {
                uint64_t slen = read_u64(f);
                _fseeki64(f, (long long)slen, SEEK_CUR);
            }
        } else if (atype < 11) {
            _fseeki64(f, (long long)(alen * sizes[atype]), SEEK_CUR);
        } else {
            _fseeki64(f, (long long)(alen * 8), SEEK_CUR);
        }
    } else if (vtype < 11) {
        fseek(f, sizes[vtype], SEEK_CUR);
    }
}

/*
 * Parse a GGUF file header and extract tensor metadata.
 * Does NOT read tensor data — just builds the offset table.
 */
static int parse_gguf(const char* path, GGUFModel* model) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;

    memset(model, 0, sizeof(GGUFModel));

    /* Magic */
    char magic[4];
    fread(magic, 1, 4, f);
    if (memcmp(magic, "GGUF", 4) != 0) { fclose(f); return -1; }

    model->version = read_u32(f);
    model->n_tensors = read_u64(f);
    uint64_t n_kv = read_u64(f);

    /* Parse KV pairs — extract model config */
    uint64_t kv;
    char key[512], sval[512];
    for (kv = 0; kv < n_kv; kv++) {
        read_gguf_string(f, key, sizeof(key));
        uint32_t vtype = read_u32(f);
        if (vtype == 4) { /* uint32 */
            uint32_t val = read_u32(f);
            if (strstr(key, "expert_count")) model->num_experts = val;
            if (strstr(key, "expert_used_count")) model->expert_used_count = val;
            if (strstr(key, "expert_feed_forward_length")) model->expert_intermediate = val;
            if (strstr(key, "feed_forward_length") && !strstr(key, "expert_feed_forward_length"))
                model->feed_forward_length = val;
            if (strstr(key, "embedding_length")) model->hidden_dim = val;
            if (strstr(key, "block_count")) model->num_layers = val;
            if (strstr(key, "head_count_kv")) model->head_count_kv = val;
            else if (strstr(key, "head_count")) model->head_count = val;
            if (strstr(key, "ssm.state_size")) model->ssm_state_size = val;
            if (strstr(key, "ssm.inner_size")) model->ssm_inner_size = val;
            if (strstr(key, "ssm.conv_kernel")) model->ssm_conv_kernel = val;
            /* Store additional useful metadata */
        } else if (vtype == 6) { /* float32 */
            float fval;
            fread(&fval, 4, 1, f);
            if (strstr(key, "rope.freq_base")) model->rope_theta = fval;
            if (strstr(key, "routed_scaling_factor")) model->routed_scaling_factor = fval;
        } else if (vtype == 9 && strstr(key, "dimension_sections")) {
            /* Read rope.dimension_sections array */
            uint32_t atype = read_u32(f);
            uint64_t alen = read_u64(f);
            fprintf(stderr, "  GGUF rope.dimension_sections: [");
            for (uint64_t ai = 0; ai < alen && ai < 16; ai++) {
                if (atype == 5) { int32_t v; fread(&v, 4, 1, f); fprintf(stderr, "%d%s", v, ai+1<alen?",":""); }
                else if (atype == 4) { uint32_t v = read_u32(f); fprintf(stderr, "%u%s", v, ai+1<alen?",":""); }
                else { skip_kv_value(f, atype); }
            }
            fprintf(stderr, "] (len=%llu, atype=%u)\n", (unsigned long long)alen, atype);
        } else {
            skip_kv_value(f, vtype);
        }
    }

    /* Parse tensor info */
    uint64_t ti;
    uint64_t n = model->n_tensors;
    if (n > MAX_TENSORS) n = MAX_TENSORS;

    for (ti = 0; ti < n; ti++) {
        TensorInfo* t = &model->tensors[ti];
        read_gguf_string(f, t->name, MAX_NAME_LEN);
        t->n_dims = read_u32(f);

        uint64_t total_elements = 1;
        int d;
        for (d = 0; d < t->n_dims && d < MAX_DIMS; d++) {
            t->dims[d] = read_u64(f);
            total_elements *= t->dims[d];
        }
        t->type = read_u32(f);
        t->offset = read_u64(f);

        /* Calculate data size */
        int bw = ggml_block_weights(t->type);
        int bs = ggml_block_size(t->type);
        if (bw > 0 && bs > 0) {
            t->data_size = (total_elements / bw) * bs;
        } else {
            t->data_size = total_elements * 4; /* fallback: assume FP32 */
        }
    }

    /* Data starts at aligned offset after header */
    model->data_start = (uint64_t)_ftelli64(f);
    /* GGUF v3 aligns data to 32 bytes */
    model->data_start = (model->data_start + 31) & ~31ULL;

    fclose(f);
    return 0;
}

/*
 * Parse a split GGUF (multiple shards).
 * Reads KV from shard 1, tensors from all shards.
 * shard_pattern: path with the shard number pattern, e.g.
 *   "D:/models/model-00001-of-00006.gguf" — auto-detects and reads all 6.
 */
static int parse_gguf_split(const char* shard1_path, GGUFModel* model) {
    /* First parse shard 1 for KV config */
    if (parse_gguf(shard1_path, model) != 0) return -1;

    /* Detect split pattern: find "00001-of-NNNNN" */
    char base[512];
    strncpy(base, shard1_path, 511); base[511] = 0;
    char* pos = strstr(base, "00001-of-");
    if (!pos) {
        /* Not a split GGUF — single file, already parsed */
        model->num_shards = 1;
        strncpy(model->shard_paths[0], shard1_path, 511);
        model->shard_data_starts[0] = model->data_start;
        return 0;
    }

    /* Extract total shards */
    int total_shards = atoi(pos + 9); /* after "00001-of-" */
    if (total_shards < 2 || total_shards > MAX_SHARDS) total_shards = 1;

    model->num_shards = total_shards;
    strncpy(model->shard_paths[0], shard1_path, 511);
    model->shard_data_starts[0] = model->data_start;

    /* Parse remaining shards for tensor info */
    int shard;
    for (shard = 2; shard <= total_shards; shard++) {
        char shard_path[512];
        strncpy(shard_path, shard1_path, 511);
        char* sp = strstr(shard_path, "00001-of-");
        if (sp) {
            /* Replace "00001" with shard number */
            char num[6];
            snprintf(num, 6, "%05d", shard);
            memcpy(sp, num, 5);
        }

        strncpy(model->shard_paths[shard - 1], shard_path, 511);

        FILE* f = fopen(shard_path, "rb");
        if (!f) continue;

        char magic[4];
        fread(magic, 1, 4, f);
        if (memcmp(magic, "GGUF", 4) != 0) { fclose(f); continue; }

        uint32_t ver = read_u32(f);
        uint64_t nt = read_u64(f);
        uint64_t nkv = read_u64(f);

        /* Skip KV pairs */
        uint64_t kvi;
        for (kvi = 0; kvi < nkv; kvi++) {
            char key[512];
            read_gguf_string(f, key, sizeof(key));
            uint32_t vtype = read_u32(f);
            skip_kv_value(f, vtype);
        }

        /* Read tensor infos */
        uint64_t ti;
        for (ti = 0; ti < nt && model->n_tensors < MAX_TENSORS; ti++) {
            TensorInfo* t = &model->tensors[model->n_tensors];
            read_gguf_string(f, t->name, MAX_NAME_LEN);
            t->n_dims = read_u32(f);

            uint64_t total_elements = 1;
            int d;
            for (d = 0; d < t->n_dims && d < MAX_DIMS; d++) {
                t->dims[d] = read_u64(f);
                total_elements *= t->dims[d];
            }
            t->type = read_u32(f);
            t->offset = read_u64(f);
            t->shard = shard - 1;  /* 0-based shard index */

            int bw = ggml_block_weights(t->type);
            int bs = ggml_block_size(t->type);
            if (bw > 0 && bs > 0) {
                t->data_size = (total_elements / bw) * bs;
            } else {
                t->data_size = total_elements * 4;
            }

            model->n_tensors++;
        }

        /* Data start for this shard */
        model->shard_data_starts[shard - 1] = (uint64_t)_ftelli64(f);
        model->shard_data_starts[shard - 1] = (model->shard_data_starts[shard - 1] + 31) & ~31ULL;

        fclose(f);
    }

    return 0;
}

/* Find a tensor by name */
static TensorInfo* find_tensor(GGUFModel* model, const char* name) {
    uint64_t i;
    for (i = 0; i < model->n_tensors && i < MAX_TENSORS; i++) {
        if (strcmp(model->tensors[i].name, name) == 0) {
            return &model->tensors[i];
        }
    }
    return NULL;
}
