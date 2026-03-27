#pragma once
/*
 * Tiered Expert Cache — VRAM hot + RAM (static LFU + adaptive LRU)
 *
 * Architecture (professor-locked):
 *   VRAM: staging ring + tiny hotset (150-300 blocks max)
 *   RAM:  75-85% static LFU + 15-25% per-sequence adaptive LRU
 *   SSD:  everything else (read via explicit I/O on miss)
 *
 * Hard user-space memory budget. No OS paging. We own every byte.
 */

#include <cstdint>
#include <cstddef>
#include <unordered_map>
#include <vector>

// Expert block identity
struct ExpertKey {
    uint16_t layer;
    uint16_t expert_id;

    bool operator==(const ExpertKey& o) const {
        return layer == o.layer && expert_id == o.expert_id;
    }
};

struct ExpertKeyHash {
    size_t operator()(const ExpertKey& k) const {
        return ((size_t)k.layer << 16) | k.expert_id;
    }
};

enum CacheTier : uint8_t {
    TIER_VRAM   = 0,   // In GPU memory — zero additional transfer needed
    TIER_RAM    = 1,   // In pinned host memory — needs H2D (0.18 ms)
    TIER_MISS   = 2,   // Not cached — needs SSD read (1.67 ms) + H2D
};

struct LookupResult {
    CacheTier tier;
    void*     data;      // Pointer to expert block data (host or device)
    int       slot_idx;  // Internal slot index (for cache management)
};

struct CacheStats {
    uint64_t total_lookups;
    uint64_t vram_hits;
    uint64_t ram_static_hits;
    uint64_t ram_adaptive_hits;
    uint64_t misses;

    double vram_hit_rate()     const { return total_lookups ? (double)vram_hits / total_lookups : 0; }
    double ram_hit_rate()      const { return total_lookups ? (double)(ram_static_hits + ram_adaptive_hits) / total_lookups : 0; }
    double total_hit_rate()    const { return total_lookups ? (double)(vram_hits + ram_static_hits + ram_adaptive_hits) / total_lookups : 0; }
    double miss_rate()         const { return total_lookups ? (double)misses / total_lookups : 0; }
};

/*
 * ExpertCache — Three-tier expert block cache
 *
 * Usage:
 *   ExpertCache cache(300, 7000, 1571, slot_size);  // vram, static_ram, adaptive_ram
 *   cache.load_static_hotset(frequency_map);         // one-time at model load
 *   ...
 *   LookupResult r = cache.lookup({layer, expert_id});
 *   if (r.tier == TIER_MISS) {
 *       // read from SSD into staging, then insert
 *       void* staging = cache.get_staging_slot();
 *       slab_reader.read_expert(layer, expert_id, staging);
 *       cache.insert_adaptive({layer, expert_id}, staging);
 *   }
 */
class ExpertCache {
public:
    // vram_blocks:     VRAM hotset capacity (150-300)
    // static_ram:      static LFU slots in RAM (75-85% of total RAM budget)
    // adaptive_ram:    adaptive LRU slots in RAM (15-25% of total RAM budget)
    // slot_size:       bytes per expert block (3,735,552 for IQ2_XXS)
    ExpertCache(int vram_blocks, int static_ram, int adaptive_ram, size_t slot_size);
    ~ExpertCache();

    // Load static LFU hotset from frequency data
    // freq_map: (layer, expert_id) → frequency count, sorted descending
    // Fills VRAM first (hottest), then static RAM
    void load_static_hotset(const std::vector<std::pair<ExpertKey, uint64_t>>& freq_sorted);

    // Provide the raw data for a static slot (called during hotset loading)
    void fill_static_slot(const ExpertKey& key, const void* data);

    // Lookup: returns tier + data pointer. O(1) via hash map.
    LookupResult lookup(const ExpertKey& key);

    // Insert into adaptive RAM tier (on cache miss after SSD read)
    // Evicts LRU if adaptive tier is full
    // data must point to slot_size bytes; contents are copied into cache
    void insert_adaptive(const ExpertKey& key, const void* data);

    // Get a temporary staging buffer (for SSD reads before insert)
    void* get_staging_buffer();

    // Stats
    CacheStats get_stats() const { return stats_; }
    void reset_stats() { stats_ = {}; }

    // Capacity info
    int vram_capacity()     const { return vram_cap_; }
    int static_capacity()   const { return static_cap_; }
    int adaptive_capacity() const { return adaptive_cap_; }
    int total_ram()         const { return static_cap_ + adaptive_cap_; }
    size_t slot_size()      const { return slot_size_; }

private:
    int    vram_cap_;
    int    static_cap_;
    int    adaptive_cap_;
    size_t slot_size_;

    // VRAM tier: flat array of expert blocks (would be GPU memory in full engine)
    // For benchmark: just host memory simulating VRAM
    uint8_t* vram_data_;

    // RAM static tier: flat array, loaded once at startup
    uint8_t* ram_static_data_;

    // RAM adaptive tier: ring of slots with LRU eviction
    uint8_t* ram_adaptive_data_;

    // Staging buffer for SSD reads
    uint8_t* staging_buf_;

    // Lookup tables: key → {tier, slot_index}
    struct CacheEntry {
        CacheTier tier;
        int       slot_idx;
    };
    std::unordered_map<ExpertKey, CacheEntry, ExpertKeyHash> lookup_map_;

    // Adaptive LRU tracking
    struct AdaptiveSlot {
        ExpertKey key;
        uint64_t  last_access;  // monotonic counter
        bool      occupied;
    };
    std::vector<AdaptiveSlot> adaptive_slots_;
    uint64_t access_counter_;
    int      adaptive_used_;

    // Find LRU slot in adaptive tier
    int find_lru_adaptive() const;

    CacheStats stats_;
};
