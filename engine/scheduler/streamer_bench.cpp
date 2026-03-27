#include "expert_streamer.h"
#include <cstdio>
#include <cstdlib>
#include <random>
#include <unordered_set>

static const int NUM_LAYERS = 60;
static const int EXPERTS_PER_LAYER = 512;
static const int K_ACTIVE = 3;

std::vector<std::vector<ExpertTask>> generate_token(
    double vram_hit_rate, double ram_hit_rate, std::mt19937& rng, int k_override = 0)
{
    int k = (k_override > 0) ? k_override : K_ACTIVE;
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::uniform_int_distribution<int> expert_dist(0, EXPERTS_PER_LAYER - 1);
    std::vector<std::vector<ExpertTask>> layers(NUM_LAYERS);

    for (int l = 0; l < NUM_LAYERS; l++) {
        std::unordered_set<int> picked;
        while ((int)picked.size() < k) picked.insert(expert_dist(rng));
        for (int eid : picked) {
            double r = dist(rng);
            Tier tier;
            if (r < vram_hit_rate) tier = Tier::VRAM;
            else if (r < vram_hit_rate + ram_hit_rate) tier = Tier::RAM;
            else tier = Tier::SSD;
            ExpertTask t; t.layer = l; t.expert_id = eid; t.tier = tier; t.load_ms = 0; t.compute_ms = 0;
            layers[l].push_back(t);
        }
    }
    return layers;
}

void run_scenario(const char* name, double vram_hit, double ram_hit, int tokens,
                  ExpertStreamer& streamer, std::mt19937& rng)
{
    double total_ms = 0, total_compute = 0, total_io = 0;
    int total_vram = 0, total_ram = 0, total_ssd = 0;

    for (int t = 0; t < tokens; t++) {
        auto tl = generate_token(vram_hit, ram_hit, rng, streamer.get_k());
        auto res = streamer.schedule_token(tl);
        total_ms += res.total_ms;
        total_compute += res.compute_ms;
        total_io += res.wasted_ms;
        total_vram += res.vram_hits;
        total_ram += res.ram_hits;
        total_ssd += res.ssd_misses;
    }

    double avg_ms = total_ms / tokens;
    printf("%-50s %7.1f %7.2f %7.1f %7.1f %5.0f %5.0f %5.0f\n",
           name, avg_ms, 1000.0 / avg_ms,
           total_compute / tokens, total_io / tokens,
           (double)total_vram / tokens, (double)total_ram / tokens,
           (double)total_ssd / tokens);
}

int main() {
    TimingConfig cfg;
    ExpertStreamer streamer(cfg);
    std::mt19937 rng(42);

    double pure_compute = cfg.num_layers * cfg.K * cfg.compute_per_expert_ms;

    printf("================================================================\n");
    printf("Expert Streaming Scheduler Benchmark\n");
    printf("================================================================\n");
    printf("SSD=%.2f ms, H2D=%.3f ms, compute=%.2f ms/expert\n",
           cfg.ssd_read_ms, cfg.h2d_ms, cfg.compute_per_expert_ms);
    printf("Pure compute floor: %.1f ms = %.2f tok/s\n\n", pure_compute, 1000.0 / pure_compute);

    printf("%-50s %8s %8s %8s %8s %5s %5s %5s\n",
           "Scenario", "ms/tok", "tok/s", "compute", "io_ovr", "VRAM", "RAM", "SSD");
    printf("--------------------------------------------------------------------------------------------\n");

    run_scenario("All VRAM (best case)",                 1.0,  0.0,  100, streamer, rng);
    run_scenario("All RAM (H2D only)",                   0.0,  1.0,  100, streamer, rng);
    run_scenario("All SSD (worst case)",                 0.0,  0.0,  100, streamer, rng);
    run_scenario("48% hit (synthetic Zipf cache)",       0.06, 0.42, 100, streamer, rng);
    run_scenario("67% hit (measured real traces)",        0.06, 0.61, 100, streamer, rng);
    run_scenario("67% hit (more VRAM)",                  0.10, 0.57, 100, streamer, rng);
    run_scenario("80% hit (warm cache)",                 0.10, 0.70, 100, streamer, rng);
    run_scenario("90% hit (very warm)",                  0.15, 0.75, 100, streamer, rng);
    run_scenario("48% hit, long seq (500 tok)",          0.06, 0.42, 500, streamer, rng);

    // CORRECTED K sweep: per-expert compute is FIXED at 1.71 ms
    // Adding more experts adds MORE total compute, not less per expert
    // Professor correction: T_layer(K) = T_attn + K * T_expert - T_overlap(K)
    printf("\n--- CORRECTED K sweep (fixed per-expert compute = 1.71 ms) ---\n");
    printf("Per-expert compute is constant. More K = more total work.\n\n");

    int k_values[] = {3, 4, 5, 6, 8, 10};
    printf("%-50s %8s %8s %8s %8s %5s %5s %5s\n",
           "Scenario", "ms/tok", "tok/s", "compute", "io_ovr", "VRAM", "RAM", "SSD");
    printf("--------------------------------------------------------------------------------------------\n");

    for (int kv : k_values) {
        TimingConfig cfgK;
        cfgK.K = kv;
        cfgK.compute_per_expert_ms = 1.71;  // FIXED per expert, not divided
        ExpertStreamer sK(cfgK);
        double pcK = cfgK.num_layers * cfgK.K * cfgK.compute_per_expert_ms;

        char name_67[80], name_ssd[80], name_vram[80];
        snprintf(name_67, 80, "K=%d 67%% hit (compute=%.0fms)", kv, pcK);
        snprintf(name_ssd, 80, "K=%d all SSD", kv);
        snprintf(name_vram, 80, "K=%d all VRAM (floor=%.0fms=%.2ftps)", kv, pcK, 1000.0/pcK);

        run_scenario(name_vram, 1.0, 0.0, 100, sK, rng);
        run_scenario(name_67, 0.06, 0.61, 100, sK, rng);
        run_scenario(name_ssd, 0.0, 0.0, 100, sK, rng);
        printf("\n");
    }

    printf("\n================================================================\n");

    auto dl = generate_token(0.06, 0.42, rng);
    auto result = streamer.schedule_token(dl);
    printf("\nDetailed: 48%% hit scenario\n");
    printf("  Total: %.1f ms (%.2f tok/s)\n", result.total_ms, 1000.0 / result.total_ms);
    printf("  Compute: %.1f ms\n", result.compute_ms);
    printf("  I/O overhead: %.1f ms (%.1f%%)\n",
           result.wasted_ms, 100.0 * result.wasted_ms / result.total_ms);
    printf("  VRAM: %d, RAM: %d, SSD: %d\n",
           result.vram_hits, result.ram_hits, result.ssd_misses);
    printf("\nDONE\n");
    return 0;
}
