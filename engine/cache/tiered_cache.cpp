#include "tiered_cache.h"
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <climits>

#ifdef _WIN32
#include <malloc.h>  // _aligned_malloc / _aligned_free
#define ALIGNED_ALLOC(size, align) _aligned_malloc(size, align)
#define ALIGNED_FREE(ptr) _aligned_free(ptr)
#else
#include <cstdlib>
#define ALIGNED_ALLOC(size, align) aligned_alloc(align, size)
#define ALIGNED_FREE(ptr) free(ptr)
#endif

static const size_t CACHE_ALIGNMENT = 65536;  // 64 KiB — matches slab slot alignment

ExpertCache::ExpertCache(int vram_blocks, int static_ram, int adaptive_ram, size_t slot_size)
    : vram_cap_(vram_blocks)
    , static_cap_(static_ram)
    , adaptive_cap_(adaptive_ram)
    , slot_size_(slot_size)
    , vram_data_(nullptr)
    , ram_static_data_(nullptr)
    , ram_adaptive_data_(nullptr)
    , staging_buf_(nullptr)
    , access_counter_(0)
    , adaptive_used_(0)
    , stats_{}
{
    // Allocate aligned memory for each tier
    if (vram_cap_ > 0) {
        vram_data_ = (uint8_t*)ALIGNED_ALLOC((size_t)vram_cap_ * slot_size_, CACHE_ALIGNMENT);
        if (vram_data_) memset(vram_data_, 0, (size_t)vram_cap_ * slot_size_);
    }

    if (static_cap_ > 0) {
        ram_static_data_ = (uint8_t*)ALIGNED_ALLOC((size_t)static_cap_ * slot_size_, CACHE_ALIGNMENT);
        if (ram_static_data_) memset(ram_static_data_, 0, (size_t)static_cap_ * slot_size_);
    }

    if (adaptive_cap_ > 0) {
        ram_adaptive_data_ = (uint8_t*)ALIGNED_ALLOC((size_t)adaptive_cap_ * slot_size_, CACHE_ALIGNMENT);
        if (ram_adaptive_data_) memset(ram_adaptive_data_, 0, (size_t)adaptive_cap_ * slot_size_);
    }

    // Staging buffer for SSD reads (one slot)
    staging_buf_ = (uint8_t*)ALIGNED_ALLOC(slot_size_, CACHE_ALIGNMENT);

    // Initialize adaptive slots
    adaptive_slots_.resize(adaptive_cap_);
    for (auto& s : adaptive_slots_) {
        s.occupied = false;
        s.last_access = 0;
    }

    // Reserve hash map capacity
    lookup_map_.reserve(vram_cap_ + static_cap_ + adaptive_cap_);
}

ExpertCache::~ExpertCache() {
    if (vram_data_)         ALIGNED_FREE(vram_data_);
    if (ram_static_data_)   ALIGNED_FREE(ram_static_data_);
    if (ram_adaptive_data_) ALIGNED_FREE(ram_adaptive_data_);
    if (staging_buf_)       ALIGNED_FREE(staging_buf_);
}

void ExpertCache::load_static_hotset(const std::vector<std::pair<ExpertKey, uint64_t>>& freq_sorted) {
    // freq_sorted must be sorted descending by frequency
    // Fill VRAM first (hottest), then static RAM
    int idx = 0;

    // VRAM tier
    for (int i = 0; i < vram_cap_ && idx < (int)freq_sorted.size(); i++, idx++) {
        const ExpertKey& key = freq_sorted[idx].first;
        CacheEntry entry;
        entry.tier = TIER_VRAM;
        entry.slot_idx = i;
        lookup_map_[key] = entry;
    }

    // Static RAM tier
    for (int i = 0; i < static_cap_ && idx < (int)freq_sorted.size(); i++, idx++) {
        const ExpertKey& key = freq_sorted[idx].first;
        CacheEntry entry;
        entry.tier = TIER_RAM;
        entry.slot_idx = i;  // index into ram_static_data_
        lookup_map_[key] = entry;
    }
}

void ExpertCache::fill_static_slot(const ExpertKey& key, const void* data) {
    auto it = lookup_map_.find(key);
    if (it == lookup_map_.end()) return;

    const CacheEntry& entry = it->second;
    if (entry.tier == TIER_VRAM) {
        memcpy(vram_data_ + (size_t)entry.slot_idx * slot_size_, data, slot_size_);
    } else if (entry.tier == TIER_RAM) {
        memcpy(ram_static_data_ + (size_t)entry.slot_idx * slot_size_, data, slot_size_);
    }
}

LookupResult ExpertCache::lookup(const ExpertKey& key) {
    stats_.total_lookups++;
    access_counter_++;

    auto it = lookup_map_.find(key);
    if (it == lookup_map_.end()) {
        stats_.misses++;
        return { TIER_MISS, nullptr, -1 };
    }

    const CacheEntry& entry = it->second;

    if (entry.tier == TIER_VRAM) {
        stats_.vram_hits++;
        return { TIER_VRAM, vram_data_ + (size_t)entry.slot_idx * slot_size_, entry.slot_idx };
    }

    // RAM tier: static (slot_idx < static_cap_) or adaptive (slot_idx >= static_cap_)
    if (entry.slot_idx < static_cap_) {
        stats_.ram_static_hits++;
        return { TIER_RAM, ram_static_data_ + (size_t)entry.slot_idx * slot_size_, entry.slot_idx };
    }

    // Adaptive slot (slot_idx is offset by static_cap_)
    int adaptive_idx = entry.slot_idx - static_cap_;
    if (adaptive_idx >= 0 && adaptive_idx < adaptive_cap_) {
        adaptive_slots_[adaptive_idx].last_access = access_counter_;
        stats_.ram_adaptive_hits++;
        return { TIER_RAM, ram_adaptive_data_ + (size_t)adaptive_idx * slot_size_, adaptive_idx };
    }

    stats_.misses++;
    return { TIER_MISS, nullptr, -1 };
}

void ExpertCache::insert_adaptive(const ExpertKey& key, const void* data) {
    // Check if already cached (promote if so)
    auto it = lookup_map_.find(key);
    if (it != lookup_map_.end()) {
        // Already in cache — just update access time if adaptive
        return;
    }

    int slot_idx;
    if (adaptive_used_ < adaptive_cap_) {
        // Free slot available
        slot_idx = adaptive_used_++;
    } else {
        // Evict LRU
        slot_idx = find_lru_adaptive();
        // Remove evicted entry from lookup map
        if (adaptive_slots_[slot_idx].occupied) {
            lookup_map_.erase(adaptive_slots_[slot_idx].key);
        }
    }

    // Copy data into adaptive slot
    memcpy(ram_adaptive_data_ + (size_t)slot_idx * slot_size_, data, slot_size_);

    // Update tracking
    adaptive_slots_[slot_idx].key = key;
    adaptive_slots_[slot_idx].last_access = access_counter_;
    adaptive_slots_[slot_idx].occupied = true;

    // Use a special encoding for adaptive entries: slot_idx stored directly,
    // but tier indicates it's adaptive via a flag approach.
    // Simpler: we use a negative offset convention — adaptive slots have
    // slot_idx in the adaptive range. We distinguish static from adaptive
    // by checking the tier at lookup time.
    CacheEntry entry;
    entry.tier = TIER_RAM;
    entry.slot_idx = slot_idx + static_cap_;  // offset past static range
    lookup_map_[key] = entry;
}

int ExpertCache::find_lru_adaptive() const {
    uint64_t min_access = UINT64_MAX;
    int min_idx = 0;
    for (int i = 0; i < adaptive_cap_; i++) {
        if (!adaptive_slots_[i].occupied) return i;  // prefer empty slots
        if (adaptive_slots_[i].last_access < min_access) {
            min_access = adaptive_slots_[i].last_access;
            min_idx = i;
        }
    }
    return min_idx;
}

void* ExpertCache::get_staging_buffer() {
    return staging_buf_;
}
