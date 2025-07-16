#ifndef Q_TABLE_OPTIMIZED_H
#define Q_TABLE_OPTIMIZED_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

// Optimized Q-table structure for improved memory access patterns
typedef struct {
    float* data;                    // Flattened 1D array for better cache locality
    int num_states;                 // Number of states
    int num_actions;                // Number of actions
    int state_stride;               // Stride for state indexing (= num_actions)
    
    // Cache optimization structures
    float* max_q_cache;             // Cache of max Q-values per state
    int* best_action_cache;         // Cache of best actions per state
    bool* cache_valid;              // Validity flags for cached values
    
    // Memory-aligned access helpers
    float* state_rows[256];         // Pre-computed row pointers (for small state spaces)
    bool use_row_cache;             // Whether to use row pointer cache
    
    // Hot path optimization
    float* last_state_ptr;          // Pointer to last accessed state
    int last_state_id;              // ID of last accessed state
    
    // SIMD optimization support
    bool simd_enabled;              // Whether SIMD operations are available
    int simd_alignment;             // Memory alignment for SIMD (16 or 32 bytes)
} OptimizedQTable;

// Memory allocation strategies
typedef enum {
    ALLOC_STANDARD,                 // Standard malloc
    ALLOC_ALIGNED,                  // Aligned allocation for SIMD
    ALLOC_HUGE_PAGES,              // Use huge pages if available
    ALLOC_NUMA_LOCAL               // NUMA-aware allocation
} QTableAllocStrategy;

// Access pattern hints for optimization
typedef struct {
    bool frequent_max_queries;      // Frequently query max Q-values
    bool sequential_state_access;   // Access states sequentially
    bool batch_updates;             // Update multiple values at once
    bool cache_friendly_training;   // Training follows cache-friendly patterns
} AccessPatternHints;

// Performance counters for monitoring
typedef struct {
    uint64_t cache_hits;            // Cache hits for max Q-values
    uint64_t cache_misses;          // Cache misses
    uint64_t total_accesses;        // Total Q-table accesses
    uint64_t batch_operations;      // Batched operations performed
    uint64_t simd_operations;       // SIMD operations used
} QTablePerfCounters;

// Optimized Q-table functions
OptimizedQTable* create_optimized_qtable(int num_states, int num_actions, 
                                         QTableAllocStrategy strategy,
                                         AccessPatternHints hints);
void destroy_optimized_qtable(OptimizedQTable* qtable);

// Fast inline access functions (defined in header for inlining)
static inline float* get_state_row_fast(OptimizedQTable* qtable, int state) {
    if (qtable->use_row_cache && state < 256) {
        return qtable->state_rows[state];
    }
    return qtable->data + (state * qtable->state_stride);
}

static inline float get_q_value_fast(OptimizedQTable* qtable, int state, int action) {
    return qtable->data[state * qtable->state_stride + action];
}

static inline void set_q_value_fast(OptimizedQTable* qtable, int state, int action, float value) {
    qtable->data[state * qtable->state_stride + action] = value;
    // Invalidate cache for this state
    if (qtable->cache_valid) {
        qtable->cache_valid[state] = false;
    }
}

// Optimized operations
float get_max_q_value_cached(OptimizedQTable* qtable, int state);
int get_best_action_cached(OptimizedQTable* qtable, int state);
void invalidate_state_cache(OptimizedQTable* qtable, int state);
void invalidate_all_caches(OptimizedQTable* qtable);

// Batch operations for better cache utilization
void batch_update_q_values(OptimizedQTable* qtable, int* states, int* actions, 
                          float* values, int count);
void batch_get_q_values(OptimizedQTable* qtable, int* states, int* actions, 
                       float* values, int count);
void batch_get_max_q_values(OptimizedQTable* qtable, int* states, 
                           float* max_values, int count);

// SIMD-optimized operations (when available)
void simd_update_state_row(OptimizedQTable* qtable, int state, float* new_values);
float simd_max_in_row(OptimizedQTable* qtable, int state);
int simd_argmax_in_row(OptimizedQTable* qtable, int state);

// Memory layout optimization
void optimize_memory_layout(OptimizedQTable* qtable, int* access_frequency);
void prefetch_state_data(OptimizedQTable* qtable, int state);
void warm_up_caches(OptimizedQTable* qtable, int* likely_states, int count);

// Performance monitoring
void reset_perf_counters(OptimizedQTable* qtable);
QTablePerfCounters get_perf_counters(OptimizedQTable* qtable);
void print_perf_stats(OptimizedQTable* qtable);
float calculate_cache_hit_ratio(OptimizedQTable* qtable);

// Compatibility layer for existing code
typedef struct {
    OptimizedQTable* qtable;
    QTablePerfCounters* counters;
} QTableWrapper;

QTableWrapper* wrap_qtable_for_agent(int num_states, int num_actions);
void destroy_qtable_wrapper(QTableWrapper* wrapper);
float qtable_get_value(QTableWrapper* wrapper, int state, int action);
void qtable_set_value(QTableWrapper* wrapper, int state, int action, float value);
int qtable_get_best_action(QTableWrapper* wrapper, int state);
float qtable_get_max_value(QTableWrapper* wrapper, int state);

// Memory-mapped Q-table for very large state spaces
typedef struct {
    OptimizedQTable base;
    void* mapped_memory;
    size_t mapped_size;
    char* filename;
    bool read_only;
} MappedQTable;

MappedQTable* create_mapped_qtable(const char* filename, int num_states, 
                                  int num_actions, bool create_new);
void destroy_mapped_qtable(MappedQTable* qtable);
bool sync_mapped_qtable(MappedQTable* qtable);

// Compression for storage efficiency
typedef struct {
    uint16_t* compressed_data;      // Quantized Q-values
    float scale_factor;             // Scaling factor for quantization
    float offset;                   // Offset for quantization
    int compression_bits;           // Bits per value (8 or 16)
} CompressedQTable;

CompressedQTable* compress_qtable(OptimizedQTable* qtable, int target_bits);
OptimizedQTable* decompress_qtable(CompressedQTable* compressed);
void destroy_compressed_qtable(CompressedQTable* qtable);

#endif // Q_TABLE_OPTIMIZED_H
