#pragma once
/*
 * Expert Streaming Scheduler — Component 4
 *
 * Professor's architecture: within-layer K-expert streaming in residency order.
 *
 * Per layer:
 *   1. Run router → get K=3 expert IDs
 *   2. Sort by residency: VRAM (ready) → RAM (need H2D) → SSD (need read + H2D)
 *   3. Start compute on first ready expert immediately
 *   4. Overlap: load next expert while computing current one
 *
 * Key ratio (professor-measured):
 *   - Cold expert load: 1.85 ms (1.67 SSD + 0.18 H2D)
 *   - Per-expert compute: 1.71 ms
 *   - Nearly equal → streaming hides most I/O behind compute
 *
 * Timing model per layer:
 *   - All 3 in VRAM: T_layer = 3 × 1.71 = 5.13 ms (pure compute)
 *   - All 3 in RAM:  T_layer = 0.18 + max(1.71, 0.18) × 2 + 1.71 = ~5.49 ms
 *   - All 3 cold:    T_layer = 1.85 + max(1.71, 1.85) × 2 + 1.71 = ~7.41 ms
 *   - Mixed (1 RAM, 1 cold, 1 VRAM): varies
 */

#include <cstdint>
#include <vector>
#include <algorithm>
#include <chrono>

// Expert residency tier
enum class Tier { VRAM, RAM, SSD };

// Per-expert scheduling entry
struct ExpertTask {
    int layer;
    int expert_id;
    Tier tier;           // where the data currently lives
    double load_ms;      // time to get data ready for compute
    double compute_ms;   // time to compute this expert's FFN
};

// Timing constants (from measurements)
struct TimingConfig {
    double ssd_read_ms;
    double h2d_ms;
    double compute_per_expert_ms;
    int    num_layers;
    int    K;

    TimingConfig() : ssd_read_ms(1.67), h2d_ms(0.184),
        compute_per_expert_ms(1.71), num_layers(60), K(3) {}
};

// Result of scheduling one token
struct TokenScheduleResult {
    double total_ms;
    double compute_ms;
    double io_ms;        // total I/O time (may overlap with compute)
    double wasted_ms;    // time GPU was idle waiting for data
    int    vram_hits;
    int    ram_hits;
    int    ssd_misses;
};

class ExpertStreamer {
public:
    ExpertStreamer(const TimingConfig& config) : cfg_(config) {}
    int get_k() const { return cfg_.K; }

    // Schedule one layer: given K expert tasks sorted by residency,
    // compute the layer time with streaming overlap
    double schedule_layer(std::vector<ExpertTask>& tasks) {
        // Sort by load time (VRAM=0, RAM=h2d, SSD=ssd+h2d)
        for (auto& t : tasks) {
            switch (t.tier) {
                case Tier::VRAM: t.load_ms = 0; break;
                case Tier::RAM:  t.load_ms = cfg_.h2d_ms; break;
                case Tier::SSD:  t.load_ms = cfg_.ssd_read_ms + cfg_.h2d_ms; break;
            }
            t.compute_ms = cfg_.compute_per_expert_ms;
        }

        // Sort: ready first (lowest load time)
        std::sort(tasks.begin(), tasks.end(),
            [](const ExpertTask& a, const ExpertTask& b) {
                return a.load_ms < b.load_ms;
            });

        // Simulate streaming pipeline
        // Timeline: load[0], then overlap load[i+1] with compute[i]
        double time = 0;

        // First expert: must wait for full load before compute starts
        time += tasks[0].load_ms;

        for (int i = 0; i < (int)tasks.size(); i++) {
            if (i > 0) {
                // Expert i's load started when expert i-1's compute started
                // So expert i is ready at: max(0, load[i] - compute[i-1])
                double overlap_deficit = tasks[i].load_ms - tasks[i-1].compute_ms;
                if (overlap_deficit > 0) {
                    time += overlap_deficit;  // GPU stall waiting for data
                }
            }
            time += tasks[i].compute_ms;
        }

        return time;
    }

    // Schedule a full token (all layers)
    TokenScheduleResult schedule_token(
        std::vector<std::vector<ExpertTask>> layer_tasks
    ) {
        TokenScheduleResult result = {};

        for (auto tasks : layer_tasks) {  // copy intentional
            for (auto& t : tasks) {
                switch (t.tier) {
                    case Tier::VRAM: result.vram_hits++; break;
                    case Tier::RAM:  result.ram_hits++; break;
                    case Tier::SSD:  result.ssd_misses++; break;
                }
            }
            result.total_ms += schedule_layer(tasks);
        }

        result.compute_ms = cfg_.num_layers * cfg_.K * cfg_.compute_per_expert_ms;
        result.io_ms = result.total_ms - result.compute_ms;
        result.wasted_ms = (result.io_ms > 0) ? result.io_ms : 0;

        return result;
    }

private:
    TimingConfig cfg_;
};
