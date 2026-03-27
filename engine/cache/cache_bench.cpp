#include "tiered_cache.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <random>

/*
 * Cache Benchmark — Test hit rates with simulated expert access patterns
 *
 * Simulates Qwen3.5-397B inference: 60 layers, 512 experts/layer, K=3
 * Uses Zipf distribution (s=0.46 from OLMoE traces) to model routing
 *
 * Professor's targets:
 *   RAM = 8,571 blocks → 67% hit rate → 2.26 tok/s with streaming
 *   75-85% static LFU + 15-25% adaptive
 */

#define NUM_LAYERS 60
#define EXPERTS_PER_LAYER 512
#define K_ACTIVE 3
#define SLOT_SIZE 3735552  // 57 * 64 KiB

// Zipf distribution generator
class ZipfGenerator {
public:
    ZipfGenerator(int n, double s, unsigned seed = 42) : n_(n), dist_(0.0, 1.0), rng_(seed) {
        // Precompute CDF
        cdf_.resize(n);
        double sum = 0;
        for (int i = 0; i < n; i++) {
            sum += 1.0 / pow(i + 1.0, s);
        }
        double cumul = 0;
        for (int i = 0; i < n; i++) {
            cumul += (1.0 / pow(i + 1.0, s)) / sum;
            cdf_[i] = cumul;
        }
    }

    int sample() {
        double u = dist_(rng_);
        auto it = std::lower_bound(cdf_.begin(), cdf_.end(), u);
        return (int)(it - cdf_.begin());
    }

private:
    int n_;
    std::vector<double> cdf_;
    std::uniform_real_distribution<double> dist_;
    std::mt19937 rng_;
};

// Build frequency map from Zipf sampling
std::vector<std::pair<ExpertKey, uint64_t>> build_frequency_map(
    int num_tokens, double zipf_s)
{
    std::unordered_map<uint32_t, uint64_t> freq;
    ZipfGenerator gen(EXPERTS_PER_LAYER, zipf_s);

    for (int t = 0; t < num_tokens; t++) {
        for (int l = 0; l < NUM_LAYERS; l++) {
            for (int k = 0; k < K_ACTIVE; k++) {
                int eid = gen.sample();
                uint32_t key = ((uint32_t)l << 16) | eid;
                freq[key]++;
            }
        }
    }

    // Convert to sorted vector
    std::vector<std::pair<ExpertKey, uint64_t>> result;
    result.reserve(freq.size());
    for (auto& [k, v] : freq) {
        ExpertKey ek;
        ek.layer = k >> 16;
        ek.expert_id = k & 0xFFFF;
        result.push_back({ek, v});
    }
    std::sort(result.begin(), result.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    return result;
}

struct BenchConfig {
    const char* name;
    int vram_blocks;
    int static_ram;
    int adaptive_ram;
    double zipf_s;
    int num_tokens;
};

void run_bench(const BenchConfig& cfg) {
    printf("\n--- %s ---\n", cfg.name);
    printf("  VRAM: %d, Static RAM: %d, Adaptive RAM: %d, Total: %d blocks\n",
           cfg.vram_blocks, cfg.static_ram, cfg.adaptive_ram,
           cfg.vram_blocks + cfg.static_ram + cfg.adaptive_ram);
    printf("  Zipf s=%.2f, Tokens=%d\n", cfg.zipf_s, cfg.num_tokens);

    // Build frequency map (use 10,000 token sample for static hotset)
    auto freq = build_frequency_map(10000, cfg.zipf_s);
    printf("  Unique expert blocks seen (10k tokens): %zu / %d\n",
           freq.size(), NUM_LAYERS * EXPERTS_PER_LAYER);

    // Create cache (use small slot for benchmark — we don't need real data)
    // Use slot_size=64 to avoid allocating 30 GB of RAM for the benchmark
    size_t bench_slot = 64;  // tiny slot for testing logic, not real data
    ExpertCache cache(cfg.vram_blocks, cfg.static_ram, cfg.adaptive_ram, bench_slot);

    // Load static hotset
    cache.load_static_hotset(freq);

    // We don't fill actual data — just testing cache hit/miss logic
    // In real engine, fill_static_slot would read from slab file

    // Simulate inference
    ZipfGenerator gen(EXPERTS_PER_LAYER, cfg.zipf_s, 12345);  // different seed than hotset
    uint8_t dummy_data[64] = {};

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int tok = 0; tok < cfg.num_tokens; tok++) {
        for (int l = 0; l < NUM_LAYERS; l++) {
            // Select K experts
            ExpertKey experts[K_ACTIVE];
            for (int k = 0; k < K_ACTIVE; k++) {
                experts[k].layer = l;
                experts[k].expert_id = gen.sample();
            }

            // Sort by residency (VRAM first, RAM second, MISS last)
            // This is the within-layer streaming order from the professor
            LookupResult results[K_ACTIVE];
            for (int k = 0; k < K_ACTIVE; k++) {
                results[k] = cache.lookup(experts[k]);
            }

            // Handle misses: insert into adaptive cache
            for (int k = 0; k < K_ACTIVE; k++) {
                if (results[k].tier == TIER_MISS) {
                    cache.insert_adaptive(experts[k], dummy_data);
                }
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    CacheStats stats = cache.get_stats();

    printf("  Results:\n");
    printf("    Total lookups:    %llu\n", (unsigned long long)stats.total_lookups);
    printf("    VRAM hits:        %llu (%.1f%%)\n",
           (unsigned long long)stats.vram_hits, stats.vram_hit_rate() * 100);
    printf("    RAM static hits:  %llu (%.1f%%)\n",
           (unsigned long long)stats.ram_static_hits,
           stats.total_lookups ? (double)stats.ram_static_hits / stats.total_lookups * 100 : 0);
    printf("    RAM adaptive hits:%llu (%.1f%%)\n",
           (unsigned long long)stats.ram_adaptive_hits,
           stats.total_lookups ? (double)stats.ram_adaptive_hits / stats.total_lookups * 100 : 0);
    printf("    Misses:           %llu (%.1f%%)\n",
           (unsigned long long)stats.misses, stats.miss_rate() * 100);
    printf("    Total hit rate:   %.1f%%\n", stats.total_hit_rate() * 100);
    printf("    Time: %.1f ms (%.0f lookups/ms)\n", ms, stats.total_lookups / ms);

    // Project tok/s
    // Per miss: 1.85 ms (SSD + H2D), per RAM hit: 0.18 ms (H2D), per VRAM hit: 0 ms
    double misses_per_token = (double)stats.misses / cfg.num_tokens;
    double ram_hits_per_token = (double)(stats.ram_static_hits + stats.ram_adaptive_hits) / cfg.num_tokens;
    double io_ms_per_token = misses_per_token * 1.85 + ram_hits_per_token * 0.18;
    double compute_ms = 427.0;  // T_c = 60 * 7.12

    double causal_ms = compute_ms + io_ms_per_token;
    double stream_ms = compute_ms;  // with streaming, I/O hidden if < compute per layer
    // More accurate: per-layer max(compute, io)
    double misses_per_layer = misses_per_token / NUM_LAYERS;
    double io_per_layer = misses_per_layer * 1.85;
    double compute_per_layer = 7.12;
    double layer_ms = (io_per_layer > compute_per_layer) ?
                      io_per_layer : compute_per_layer;
    double streamed_token_ms = layer_ms * NUM_LAYERS;

    printf("  Projected tok/s:\n");
    printf("    Causal (no overlap): %.1f ms/tok = %.2f tok/s\n",
           causal_ms, 1000.0 / causal_ms);
    printf("    Streamed (within-layer): %.1f ms/tok = %.2f tok/s\n",
           streamed_token_ms, 1000.0 / streamed_token_ms);
}

int main() {
    printf("================================================================\n");
    printf("Tiered Expert Cache Benchmark\n");
    printf("================================================================\n");
    printf("Model: Qwen3.5-397B-A17B IQ2_XXS\n");
    printf("Layers: %d, Experts/layer: %d, K: %d\n", NUM_LAYERS, EXPERTS_PER_LAYER, K_ACTIVE);
    printf("Slot size: %d bytes (benchmark uses 64B proxy)\n", SLOT_SIZE);

    // Professor's recommended config: 8,571 total RAM blocks
    // 80% static (6,857) + 20% adaptive (1,714)
    BenchConfig configs[] = {
        {"Prof config: 300V + 6857S + 1714A, Zipf=0.46",
         300, 6857, 1714, 0.46, 500},
        {"Prof config: 150V + 7000S + 1571A, Zipf=0.46",
         150, 7000, 1571, 0.46, 500},
        {"No VRAM: 0V + 7000S + 1571A, Zipf=0.46",
         0, 7000, 1571, 0.46, 500},
        {"Small cache: 0V + 3000S + 1000A, Zipf=0.46",
         0, 3000, 1000, 0.46, 500},
        {"Prof config, higher skew: 300V + 6857S + 1714A, Zipf=0.60",
         300, 6857, 1714, 0.60, 500},
        {"Prof config, lower skew: 300V + 6857S + 1714A, Zipf=0.30",
         300, 6857, 1714, 0.30, 500},
        {"Long sequence: 300V + 6857S + 1714A, Zipf=0.46, 2000 tokens",
         300, 6857, 1714, 0.46, 2000},
    };

    for (auto& cfg : configs) {
        run_bench(cfg);
    }

    printf("\n================================================================\n");
    printf("DONE\n");
    return 0;
}
