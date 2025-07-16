#include "q_table_optimized.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#ifdef __SSE2__
#include <emmintrin.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

// Platform-specific memory alignment
#ifdef _WIN32
#include <malloc.h>
#define aligned_alloc(alignment, size) _aligned_malloc(size, alignment)
#define aligned_free _aligned_free
#else
#include <stdlib.h>
#define aligned_free free
#endif

// Performance counters (thread-local for multi-threading support)
static __thread QTablePerfCounters g_perf_counters = {0};

// Create optimized Q-table with specified allocation strategy
OptimizedQTable* create_optimized_qtable(int num_states, int num_actions, 
                                         QTableAllocStrategy strategy,
                                         AccessPatternHints hints) {
    if (num_states <= 0 || num_actions <= 0) {
        fprintf(stderr, "Error: Invalid Q-table dimensions\n");
        return NULL;
    }

    OptimizedQTable* qtable = (OptimizedQTable*)malloc(sizeof(OptimizedQTable));
    if (!qtable) {
        fprintf(stderr, "Error: Failed to allocate OptimizedQTable structure\n");
        return NULL;
    }

    qtable->num_states = num_states;
    qtable->num_actions = num_actions;
    qtable->state_stride = num_actions;  // Stride for row-major layout
    qtable->last_state_id = -1;
    qtable->last_state_ptr = NULL;

    // Determine SIMD capabilities and alignment
    qtable->simd_enabled = false;
    qtable->simd_alignment = 16;  // Default to SSE alignment

#ifdef __AVX2__
    qtable->simd_enabled = true;
    qtable->simd_alignment = 32;
#elif defined(__SSE2__)
    qtable->simd_enabled = true;
    qtable->simd_alignment = 16;
#endif

    // Calculate total memory needed
    size_t data_size = num_states * num_actions * sizeof(float);
    
    // Allocate main data array based on strategy
    switch (strategy) {
        case ALLOC_ALIGNED:
            qtable->data = (float*)aligned_alloc(qtable->simd_alignment, data_size);
            break;
        case ALLOC_STANDARD:
        default:
            qtable->data = (float*)malloc(data_size);
            break;
    }

    if (!qtable->data) {
        fprintf(stderr, "Error: Failed to allocate Q-table data array\n");
        free(qtable);
        return NULL;
    }

    // Initialize data to zero
    memset(qtable->data, 0, data_size);

    // Allocate cache structures if frequent max queries are expected
    if (hints.frequent_max_queries) {
        qtable->max_q_cache = (float*)malloc(num_states * sizeof(float));
        qtable->best_action_cache = (int*)malloc(num_states * sizeof(int));
        qtable->cache_valid = (bool*)calloc(num_states, sizeof(bool));

        if (!qtable->max_q_cache || !qtable->best_action_cache || !qtable->cache_valid) {
            fprintf(stderr, "Error: Failed to allocate cache structures\n");
            destroy_optimized_qtable(qtable);
            return NULL;
        }
    } else {
        qtable->max_q_cache = NULL;
        qtable->best_action_cache = NULL;
        qtable->cache_valid = NULL;
    }

    // Setup row pointer cache for small state spaces
    qtable->use_row_cache = (num_states <= 256);
    if (qtable->use_row_cache) {
        for (int i = 0; i < num_states && i < 256; i++) {
            qtable->state_rows[i] = qtable->data + (i * qtable->state_stride);
        }
    }

    printf("Created optimized Q-table: %dx%d, SIMD: %s, Cache: %s, RowCache: %s\n",
           num_states, num_actions,
           qtable->simd_enabled ? "enabled" : "disabled",
           qtable->max_q_cache ? "enabled" : "disabled",
           qtable->use_row_cache ? "enabled" : "disabled");

    return qtable;
}

// Destroy optimized Q-table
void destroy_optimized_qtable(OptimizedQTable* qtable) {
    if (!qtable) return;

    if (qtable->data) {
        if (qtable->simd_alignment > 0) {
            aligned_free(qtable->data);
        } else {
            free(qtable->data);
        }
    }

    free(qtable->max_q_cache);
    free(qtable->best_action_cache);
    free(qtable->cache_valid);
    free(qtable);
}

// Get max Q-value for a state with caching
float get_max_q_value_cached(OptimizedQTable* qtable, int state) {
    if (!qtable || state < 0 || state >= qtable->num_states) {
        return 0.0f;
    }

    g_perf_counters.total_accesses++;

    // Check cache if available
    if (qtable->cache_valid && qtable->cache_valid[state]) {
        g_perf_counters.cache_hits++;
        return qtable->max_q_cache[state];
    }

    g_perf_counters.cache_misses++;

    // Calculate max Q-value
    float* state_data = get_state_row_fast(qtable, state);
    float max_q = state_data[0];

    // Use SIMD if available and beneficial
    if (qtable->simd_enabled && qtable->num_actions >= 8) {
        max_q = simd_max_in_row(qtable, state);
    } else {
        // Standard loop
        for (int a = 1; a < qtable->num_actions; a++) {
            if (state_data[a] > max_q) {
                max_q = state_data[a];
            }
        }
    }

    // Cache the result
    if (qtable->max_q_cache) {
        qtable->max_q_cache[state] = max_q;
        qtable->cache_valid[state] = true;
    }

    return max_q;
}

// Get best action for a state with caching
int get_best_action_cached(OptimizedQTable* qtable, int state) {
    if (!qtable || state < 0 || state >= qtable->num_states) {
        return 0;
    }

    g_perf_counters.total_accesses++;

    // Check cache if available
    if (qtable->cache_valid && qtable->cache_valid[state]) {
        g_perf_counters.cache_hits++;
        return qtable->best_action_cache[state];
    }

    g_perf_counters.cache_misses++;

    // Calculate best action
    float* state_data = get_state_row_fast(qtable, state);
    int best_action = 0;
    float max_q = state_data[0];

    // Use SIMD if available and beneficial
    if (qtable->simd_enabled && qtable->num_actions >= 8) {
        best_action = simd_argmax_in_row(qtable, state);
        max_q = state_data[best_action];
    } else {
        // Standard loop
        for (int a = 1; a < qtable->num_actions; a++) {
            if (state_data[a] > max_q) {
                max_q = state_data[a];
                best_action = a;
            }
        }
    }

    // Cache the results
    if (qtable->best_action_cache && qtable->max_q_cache) {
        qtable->best_action_cache[state] = best_action;
        qtable->max_q_cache[state] = max_q;
        qtable->cache_valid[state] = true;
    }

    return best_action;
}

// Invalidate cache for a specific state
void invalidate_state_cache(OptimizedQTable* qtable, int state) {
    if (!qtable || !qtable->cache_valid || state < 0 || state >= qtable->num_states) {
        return;
    }
    qtable->cache_valid[state] = false;
}

// Invalidate all caches
void invalidate_all_caches(OptimizedQTable* qtable) {
    if (!qtable || !qtable->cache_valid) return;
    
    memset(qtable->cache_valid, false, qtable->num_states * sizeof(bool));
}

// Batch update Q-values for better cache utilization
void batch_update_q_values(OptimizedQTable* qtable, int* states, int* actions, 
                          float* values, int count) {
    if (!qtable || !states || !actions || !values) return;

    g_perf_counters.batch_operations++;

    for (int i = 0; i < count; i++) {
        int state = states[i];
        int action = actions[i];
        
        if (state >= 0 && state < qtable->num_states && 
            action >= 0 && action < qtable->num_actions) {
            set_q_value_fast(qtable, state, action, values[i]);
        }
    }
}

// Batch get Q-values
void batch_get_q_values(OptimizedQTable* qtable, int* states, int* actions, 
                       float* values, int count) {
    if (!qtable || !states || !actions || !values) return;

    g_perf_counters.batch_operations++;

    for (int i = 0; i < count; i++) {
        int state = states[i];
        int action = actions[i];
        
        if (state >= 0 && state < qtable->num_states && 
            action >= 0 && action < qtable->num_actions) {
            values[i] = get_q_value_fast(qtable, state, action);
        } else {
            values[i] = 0.0f;
        }
    }
}

// Batch get max Q-values
void batch_get_max_q_values(OptimizedQTable* qtable, int* states, 
                           float* max_values, int count) {
    if (!qtable || !states || !max_values) return;

    g_perf_counters.batch_operations++;

    for (int i = 0; i < count; i++) {
        max_values[i] = get_max_q_value_cached(qtable, states[i]);
    }
}

// SIMD-optimized operations
#ifdef __AVX2__
float simd_max_in_row(OptimizedQTable* qtable, int state) {
    if (!qtable || state < 0 || state >= qtable->num_states) {
        return 0.0f;
    }

    g_perf_counters.simd_operations++;

    float* state_data = get_state_row_fast(qtable, state);
    int num_actions = qtable->num_actions;
    
    // Process 8 floats at a time with AVX2
    __m256 max_vec = _mm256_load_ps(state_data);
    
    int simd_end = (num_actions / 8) * 8;
    for (int i = 8; i < simd_end; i += 8) {
        __m256 data_vec = _mm256_load_ps(&state_data[i]);
        max_vec = _mm256_max_ps(max_vec, data_vec);
    }
    
    // Horizontal max within the vector
    __m128 max_high = _mm256_extractf128_ps(max_vec, 1);
    __m128 max_low = _mm256_castps256_ps128(max_vec);
    __m128 max_final = _mm_max_ps(max_high, max_low);
    
    // Further reduce to single value
    max_final = _mm_max_ps(max_final, _mm_shuffle_ps(max_final, max_final, _MM_SHUFFLE(2, 3, 0, 1)));
    max_final = _mm_max_ps(max_final, _mm_shuffle_ps(max_final, max_final, _MM_SHUFFLE(1, 0, 3, 2)));
    
    float result = _mm_cvtss_f32(max_final);
    
    // Handle remaining elements
    for (int i = simd_end; i < num_actions; i++) {
        if (state_data[i] > result) {
            result = state_data[i];
        }
    }
    
    return result;
}
#elif defined(__SSE2__)
float simd_max_in_row(OptimizedQTable* qtable, int state) {
    if (!qtable || state < 0 || state >= qtable->num_states) {
        return 0.0f;
    }

    g_perf_counters.simd_operations++;

    float* state_data = get_state_row_fast(qtable, state);
    int num_actions = qtable->num_actions;
    
    // Process 4 floats at a time with SSE2
    __m128 max_vec = _mm_load_ps(state_data);
    
    int simd_end = (num_actions / 4) * 4;
    for (int i = 4; i < simd_end; i += 4) {
        __m128 data_vec = _mm_load_ps(&state_data[i]);
        max_vec = _mm_max_ps(max_vec, data_vec);
    }
    
    // Horizontal max within the vector
    max_vec = _mm_max_ps(max_vec, _mm_shuffle_ps(max_vec, max_vec, _MM_SHUFFLE(2, 3, 0, 1)));
    max_vec = _mm_max_ps(max_vec, _mm_shuffle_ps(max_vec, max_vec, _MM_SHUFFLE(1, 0, 3, 2)));
    
    float result = _mm_cvtss_f32(max_vec);
    
    // Handle remaining elements
    for (int i = simd_end; i < num_actions; i++) {
        if (state_data[i] > result) {
            result = state_data[i];
        }
    }
    
    return result;
}
#else
float simd_max_in_row(OptimizedQTable* qtable, int state) {
    // Fallback to standard implementation
    float* state_data = get_state_row_fast(qtable, state);
    float max_q = state_data[0];
    
    for (int a = 1; a < qtable->num_actions; a++) {
        if (state_data[a] > max_q) {
            max_q = state_data[a];
        }
    }
    
    return max_q;
}
#endif

// SIMD argmax implementation
int simd_argmax_in_row(OptimizedQTable* qtable, int state) {
    if (!qtable || state < 0 || state >= qtable->num_states) {
        return 0;
    }

    // For now, use standard implementation (SIMD argmax is more complex)
    float* state_data = get_state_row_fast(qtable, state);
    int best_action = 0;
    float max_q = state_data[0];
    
    for (int a = 1; a < qtable->num_actions; a++) {
        if (state_data[a] > max_q) {
            max_q = state_data[a];
            best_action = a;
        }
    }
    
    return best_action;
}

// Update entire state row with SIMD
void simd_update_state_row(OptimizedQTable* qtable, int state, float* new_values) {
    if (!qtable || state < 0 || state >= qtable->num_states || !new_values) {
        return;
    }

    float* state_data = get_state_row_fast(qtable, state);
    
#ifdef __AVX2__
    if (qtable->simd_enabled && qtable->num_actions >= 8) {
        g_perf_counters.simd_operations++;
        
        int simd_end = (qtable->num_actions / 8) * 8;
        for (int i = 0; i < simd_end; i += 8) {
            __m256 new_vec = _mm256_load_ps(&new_values[i]);
            _mm256_store_ps(&state_data[i], new_vec);
        }
        
        // Handle remaining elements
        for (int i = simd_end; i < qtable->num_actions; i++) {
            state_data[i] = new_values[i];
        }
    } else
#endif
    {
        memcpy(state_data, new_values, qtable->num_actions * sizeof(float));
    }
    
    // Invalidate cache for this state
    invalidate_state_cache(qtable, state);
}

// Memory prefetching
void prefetch_state_data(OptimizedQTable* qtable, int state) {
    if (!qtable || state < 0 || state >= qtable->num_states) {
        return;
    }

    float* state_data = get_state_row_fast(qtable, state);
    
#ifdef __builtin_prefetch
    __builtin_prefetch(state_data, 0, 3);  // Prefetch for read, high temporal locality
#endif
}

// Warm up caches with likely states
void warm_up_caches(OptimizedQTable* qtable, int* likely_states, int count) {
    if (!qtable || !likely_states) return;

    for (int i = 0; i < count; i++) {
        int state = likely_states[i];
        if (state >= 0 && state < qtable->num_states) {
            prefetch_state_data(qtable, state);
            // Force cache population
            get_max_q_value_cached(qtable, state);
            get_best_action_cached(qtable, state);
        }
    }
}

// Performance monitoring functions
void reset_perf_counters(OptimizedQTable* qtable) {
    (void)qtable;  // Unused parameter
    memset(&g_perf_counters, 0, sizeof(QTablePerfCounters));
}

QTablePerfCounters get_perf_counters(OptimizedQTable* qtable) {
    (void)qtable;  // Unused parameter
    return g_perf_counters;
}

void print_perf_stats(OptimizedQTable* qtable) {
    (void)qtable;  // Unused parameter
    
    printf("\n=== Q-Table Performance Statistics ===\n");
    printf("Total accesses: %llu\n", g_perf_counters.total_accesses);
    printf("Cache hits: %llu\n", g_perf_counters.cache_hits);
    printf("Cache misses: %llu\n", g_perf_counters.cache_misses);
    printf("Cache hit ratio: %.2f%%\n", calculate_cache_hit_ratio(qtable));
    printf("Batch operations: %llu\n", g_perf_counters.batch_operations);
    printf("SIMD operations: %llu\n", g_perf_counters.simd_operations);
    printf("=====================================\n");
}

float calculate_cache_hit_ratio(OptimizedQTable* qtable) {
    (void)qtable;  // Unused parameter
    
    uint64_t total_cache_accesses = g_perf_counters.cache_hits + g_perf_counters.cache_misses;
    if (total_cache_accesses == 0) return 0.0f;
    
    return (float)g_perf_counters.cache_hits / total_cache_accesses * 100.0f;
}

// Compatibility wrapper for existing agent code
QTableWrapper* wrap_qtable_for_agent(int num_states, int num_actions) {
    QTableWrapper* wrapper = (QTableWrapper*)malloc(sizeof(QTableWrapper));
    if (!wrapper) {
        fprintf(stderr, "Error: Failed to allocate QTableWrapper\n");
        return NULL;
    }

    // Create optimized Q-table with default settings
    AccessPatternHints hints = {
        .frequent_max_queries = true,
        .sequential_state_access = false,
        .batch_updates = false,
        .cache_friendly_training = true
    };

    wrapper->qtable = create_optimized_qtable(num_states, num_actions, ALLOC_ALIGNED, hints);
    if (!wrapper->qtable) {
        free(wrapper);
        return NULL;
    }

    wrapper->counters = (QTablePerfCounters*)malloc(sizeof(QTablePerfCounters));
    if (!wrapper->counters) {
        destroy_optimized_qtable(wrapper->qtable);
        free(wrapper);
        return NULL;
    }

    memset(wrapper->counters, 0, sizeof(QTablePerfCounters));
    return wrapper;
}

void destroy_qtable_wrapper(QTableWrapper* wrapper) {
    if (!wrapper) return;
    
    destroy_optimized_qtable(wrapper->qtable);
    free(wrapper->counters);
    free(wrapper);
}

float qtable_get_value(QTableWrapper* wrapper, int state, int action) {
    if (!wrapper || !wrapper->qtable) return 0.0f;
    
    return get_q_value_fast(wrapper->qtable, state, action);
}

void qtable_set_value(QTableWrapper* wrapper, int state, int action, float value) {
    if (!wrapper || !wrapper->qtable) return;
    
    set_q_value_fast(wrapper->qtable, state, action, value);
}

int qtable_get_best_action(QTableWrapper* wrapper, int state) {
    if (!wrapper || !wrapper->qtable) return 0;
    
    return get_best_action_cached(wrapper->qtable, state);
}

float qtable_get_max_value(QTableWrapper* wrapper, int state) {
    if (!wrapper || !wrapper->qtable) return 0.0f;
    
    return get_max_q_value_cached(wrapper->qtable, state);
}
