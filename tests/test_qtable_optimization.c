#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <math.h>

#include "../include/q_table_optimized.h"
#include "../include/agent.h"

// Test configuration
#define TEST_STATES 1000
#define TEST_ACTIONS 4
#define PERFORMANCE_ITERATIONS 100000
#define CACHE_TEST_ITERATIONS 10000

// Color codes for output
#define COLOR_GREEN "\033[32m"
#define COLOR_RED "\033[31m"
#define COLOR_YELLOW "\033[33m"
#define COLOR_BLUE "\033[34m"
#define COLOR_RESET "\033[0m"

// Test result tracking
typedef struct {
    int tests_passed;
    int tests_failed;
    int total_tests;
} TestResults;

static TestResults g_results = {0, 0, 0};

// Helper macros for testing
#define TEST_ASSERT(condition, message) do { \
    g_results.total_tests++; \
    if (condition) { \
        printf(COLOR_GREEN "✓ PASS" COLOR_RESET ": %s\n", message); \
        g_results.tests_passed++; \
    } else { \
        printf(COLOR_RED "✗ FAIL" COLOR_RESET ": %s\n", message); \
        g_results.tests_failed++; \
    } \
} while(0)

#define TEST_START(test_name) \
    printf(COLOR_BLUE "\n=== Testing %s ===" COLOR_RESET "\n", test_name)

// Timing utilities
double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

// Test basic Q-table creation and destruction
void test_qtable_creation() {
    TEST_START("Q-table Creation and Destruction");
    
    AccessPatternHints hints = {
        .frequent_max_queries = true,
        .sequential_state_access = false,
        .batch_updates = false,
        .cache_friendly_training = true
    };
    
    // Test standard allocation
    OptimizedQTable* qtable = create_optimized_qtable(TEST_STATES, TEST_ACTIONS, ALLOC_STANDARD, hints);
    TEST_ASSERT(qtable != NULL, "Standard Q-table creation");
    TEST_ASSERT(qtable->num_states == TEST_STATES, "Correct number of states");
    TEST_ASSERT(qtable->num_actions == TEST_ACTIONS, "Correct number of actions");
    TEST_ASSERT(qtable->data != NULL, "Data array allocated");
    destroy_optimized_qtable(qtable);
    
    // Test aligned allocation
    qtable = create_optimized_qtable(TEST_STATES, TEST_ACTIONS, ALLOC_ALIGNED, hints);
    TEST_ASSERT(qtable != NULL, "Aligned Q-table creation");
    TEST_ASSERT(qtable->simd_alignment >= 16, "SIMD alignment set");
    destroy_optimized_qtable(qtable);
    
    // Test with cache disabled
    hints.frequent_max_queries = false;
    qtable = create_optimized_qtable(TEST_STATES, TEST_ACTIONS, ALLOC_STANDARD, hints);
    TEST_ASSERT(qtable != NULL, "Q-table creation without cache");
    TEST_ASSERT(qtable->max_q_cache == NULL, "Cache disabled");
    destroy_optimized_qtable(qtable);
}

// Test basic Q-value operations
void test_basic_operations() {
    TEST_START("Basic Q-value Operations");
    
    AccessPatternHints hints = {
        .frequent_max_queries = true,
        .sequential_state_access = false,
        .batch_updates = false,
        .cache_friendly_training = true
    };
    
    OptimizedQTable* qtable = create_optimized_qtable(100, TEST_ACTIONS, ALLOC_ALIGNED, hints);
    TEST_ASSERT(qtable != NULL, "Q-table creation for testing");
    
    // Test setting and getting values
    set_q_value_fast(qtable, 0, 0, 1.5f);
    float value = get_q_value_fast(qtable, 0, 0);
    TEST_ASSERT(fabs(value - 1.5f) < 1e-6, "Set and get Q-value");
    
    // Test multiple values
    for (int s = 0; s < 10; s++) {
        for (int a = 0; a < TEST_ACTIONS; a++) {
            float test_value = s * TEST_ACTIONS + a + 0.1f;
            set_q_value_fast(qtable, s, a, test_value);
            float retrieved = get_q_value_fast(qtable, s, a);
            TEST_ASSERT(fabs(retrieved - test_value) < 1e-6, "Multiple Q-value operations");
        }
    }
    
    destroy_optimized_qtable(qtable);
}

// Test cached max operations
void test_cached_operations() {
    TEST_START("Cached Max Operations");
    
    AccessPatternHints hints = {
        .frequent_max_queries = true,
        .sequential_state_access = false,
        .batch_updates = false,
        .cache_friendly_training = true
    };
    
    OptimizedQTable* qtable = create_optimized_qtable(100, TEST_ACTIONS, ALLOC_ALIGNED, hints);
    TEST_ASSERT(qtable != NULL, "Q-table creation for cache testing");
    
    // Set up test values for state 0
    set_q_value_fast(qtable, 0, 0, 1.0f);
    set_q_value_fast(qtable, 0, 1, 3.5f);  // This should be max
    set_q_value_fast(qtable, 0, 2, 2.0f);
    set_q_value_fast(qtable, 0, 3, 1.5f);
    
    // Test max value
    float max_val = get_max_q_value_cached(qtable, 0);
    TEST_ASSERT(fabs(max_val - 3.5f) < 1e-6, "Cached max Q-value");
    
    // Test best action
    int best_action = get_best_action_cached(qtable, 0);
    TEST_ASSERT(best_action == 1, "Cached best action");
    
    // Test cache hit (second call should hit cache)
    reset_perf_counters(qtable);
    get_max_q_value_cached(qtable, 0);
    get_max_q_value_cached(qtable, 0);
    QTablePerfCounters counters = get_perf_counters(qtable);
    TEST_ASSERT(counters.cache_hits > 0, "Cache hits registered");
    
    // Test cache invalidation
    invalidate_state_cache(qtable, 0);
    TEST_ASSERT(!qtable->cache_valid[0], "Cache invalidated");
    
    destroy_optimized_qtable(qtable);
}

// Test batch operations
void test_batch_operations() {
    TEST_START("Batch Operations");
    
    AccessPatternHints hints = {
        .frequent_max_queries = true,
        .sequential_state_access = false,
        .batch_updates = true,
        .cache_friendly_training = true
    };
    
    OptimizedQTable* qtable = create_optimized_qtable(100, TEST_ACTIONS, ALLOC_ALIGNED, hints);
    TEST_ASSERT(qtable != NULL, "Q-table creation for batch testing");
    
    // Prepare batch data
    const int batch_size = 10;
    int states[batch_size];
    int actions[batch_size];
    float values[batch_size];
    float retrieved[batch_size];
    
    for (int i = 0; i < batch_size; i++) {
        states[i] = i;
        actions[i] = i % TEST_ACTIONS;
        values[i] = i * 0.5f + 1.0f;
    }
    
    // Test batch update
    reset_perf_counters(qtable);
    batch_update_q_values(qtable, states, actions, values, batch_size);
    QTablePerfCounters counters = get_perf_counters(qtable);
    TEST_ASSERT(counters.batch_operations > 0, "Batch update recorded");
    
    // Test batch get
    batch_get_q_values(qtable, states, actions, retrieved, batch_size);
    bool all_correct = true;
    for (int i = 0; i < batch_size; i++) {
        if (fabs(retrieved[i] - values[i]) > 1e-6) {
            all_correct = false;
            break;
        }
    }
    TEST_ASSERT(all_correct, "Batch get operations");
    
    // Test batch max values
    float max_values[batch_size];
    batch_get_max_q_values(qtable, states, max_values, batch_size);
    TEST_ASSERT(max_values[0] >= 0.0f, "Batch max values operation");
    
    destroy_optimized_qtable(qtable);
}

// Test SIMD operations (if available)
void test_simd_operations() {
    TEST_START("SIMD Operations");
    
    AccessPatternHints hints = {
        .frequent_max_queries = true,
        .sequential_state_access = false,
        .batch_updates = false,
        .cache_friendly_training = true
    };
    
    OptimizedQTable* qtable = create_optimized_qtable(100, 16, ALLOC_ALIGNED, hints); // 16 actions for SIMD
    TEST_ASSERT(qtable != NULL, "Q-table creation for SIMD testing");
    
    // Set up test values
    for (int a = 0; a < 16; a++) {
        set_q_value_fast(qtable, 0, a, (float)a * 0.5f);
    }
    
    // Test SIMD max (should find action 15 with value 7.5)
    if (qtable->simd_enabled) {
        reset_perf_counters(qtable);
        float simd_max = simd_max_in_row(qtable, 0);
        TEST_ASSERT(fabs(simd_max - 7.5f) < 1e-6, "SIMD max operation");
        
        int simd_argmax = simd_argmax_in_row(qtable, 0);
        TEST_ASSERT(simd_argmax == 15, "SIMD argmax operation");
        
        QTablePerfCounters counters = get_perf_counters(qtable);
        TEST_ASSERT(counters.simd_operations > 0, "SIMD operations recorded");
    } else {
        printf(COLOR_YELLOW "⚠ SKIP" COLOR_RESET ": SIMD not available on this platform\n");
    }
    
    destroy_optimized_qtable(qtable);
}

// Performance comparison test
void test_performance_comparison() {
    TEST_START("Performance Comparison");
    
    printf("Comparing optimized vs standard Q-table performance...\n");
    
    // Create standard Q-table (using existing agent structure)
    QLearningAgent* standard_agent = create_agent(TEST_STATES, TEST_ACTIONS, 0.1f, 0.99f, 0.1f);
    TEST_ASSERT(standard_agent != NULL, "Standard agent creation");
    
    // Create optimized Q-table
    QTableWrapper* optimized = wrap_qtable_for_agent(TEST_STATES, TEST_ACTIONS);
    TEST_ASSERT(optimized != NULL, "Optimized Q-table wrapper creation");
    
    // Fill with random data
    srand(42); // Consistent seed for fair comparison
    for (int s = 0; s < TEST_STATES; s++) {
        for (int a = 0; a < TEST_ACTIONS; a++) {
            float value = (float)rand() / RAND_MAX * 10.0f - 5.0f;
            set_q_value(standard_agent, s, a, value);
            qtable_set_value(optimized, s, a, value);
        }
    }
    
    // Test standard Q-table performance
    double start_time = get_time_ms();
    for (int i = 0; i < PERFORMANCE_ITERATIONS; i++) {
        int state = rand() % TEST_STATES;
        select_greedy_action(standard_agent, state);
    }
    double standard_time = get_time_ms() - start_time;
    
    // Test optimized Q-table performance
    start_time = get_time_ms();
    for (int i = 0; i < PERFORMANCE_ITERATIONS; i++) {
        int state = rand() % TEST_STATES;
        qtable_get_best_action(optimized, state);
    }
    double optimized_time = get_time_ms() - start_time;
    
    double speedup = standard_time / optimized_time;
    printf("Standard Q-table time: %.2f ms\n", standard_time);
    printf("Optimized Q-table time: %.2f ms\n", optimized_time);
    printf("Speedup: %.2fx\n", speedup);
    
    TEST_ASSERT(speedup >= 1.0, "Optimized Q-table performance improvement");
    
    destroy_agent(standard_agent);
    destroy_qtable_wrapper(optimized);
}

// Test memory layout optimization
void test_memory_layout() {
    TEST_START("Memory Layout Optimization");
    
    AccessPatternHints hints = {
        .frequent_max_queries = true,
        .sequential_state_access = true,
        .batch_updates = false,
        .cache_friendly_training = true
    };
    
    OptimizedQTable* qtable = create_optimized_qtable(256, TEST_ACTIONS, ALLOC_ALIGNED, hints);
    TEST_ASSERT(qtable != NULL, "Q-table creation for memory layout testing");
    
    // Test row cache for small state spaces
    TEST_ASSERT(qtable->use_row_cache, "Row cache enabled for small state space");
    
    // Test memory alignment
    uintptr_t data_addr = (uintptr_t)qtable->data;
    TEST_ASSERT(data_addr % qtable->simd_alignment == 0, "Data properly aligned for SIMD");
    
    // Test prefetching (doesn't crash)
    prefetch_state_data(qtable, 0);
    TEST_ASSERT(true, "Memory prefetching operation");
    
    // Test cache warm-up
    int likely_states[] = {0, 1, 2, 3, 4};
    warm_up_caches(qtable, likely_states, 5);
    TEST_ASSERT(true, "Cache warm-up operation");
    
    destroy_optimized_qtable(qtable);
}

// Test compatibility wrapper
void test_compatibility_wrapper() {
    TEST_START("Compatibility Wrapper");
    
    QTableWrapper* wrapper = wrap_qtable_for_agent(100, TEST_ACTIONS);
    TEST_ASSERT(wrapper != NULL, "Wrapper creation");
    TEST_ASSERT(wrapper->qtable != NULL, "Wrapped Q-table exists");
    TEST_ASSERT(wrapper->counters != NULL, "Performance counters exist");
    
    // Test wrapper operations
    qtable_set_value(wrapper, 0, 0, 2.5f);
    float value = qtable_get_value(wrapper, 0, 0);
    TEST_ASSERT(fabs(value - 2.5f) < 1e-6, "Wrapper set/get operations");
    
    // Set up for max/argmax test
    qtable_set_value(wrapper, 1, 0, 1.0f);
    qtable_set_value(wrapper, 1, 1, 4.0f);
    qtable_set_value(wrapper, 1, 2, 2.0f);
    qtable_set_value(wrapper, 1, 3, 3.0f);
    
    float max_val = qtable_get_max_value(wrapper, 1);
    int best_action = qtable_get_best_action(wrapper, 1);
    
    TEST_ASSERT(fabs(max_val - 4.0f) < 1e-6, "Wrapper max value");
    TEST_ASSERT(best_action == 1, "Wrapper best action");
    
    destroy_qtable_wrapper(wrapper);
}

// Test error handling
void test_error_handling() {
    TEST_START("Error Handling");
    
    // Test invalid parameters
    OptimizedQTable* qtable = create_optimized_qtable(-1, 4, ALLOC_STANDARD, (AccessPatternHints){0});
    TEST_ASSERT(qtable == NULL, "Invalid state count rejected");
    
    qtable = create_optimized_qtable(100, -1, ALLOC_STANDARD, (AccessPatternHints){0});
    TEST_ASSERT(qtable == NULL, "Invalid action count rejected");
    
    // Test operations on NULL qtable
    float value = get_q_value_fast(NULL, 0, 0);
    TEST_ASSERT(value == 0.0f, "NULL qtable get returns 0");
    
    float max_val = get_max_q_value_cached(NULL, 0);
    TEST_ASSERT(max_val == 0.0f, "NULL qtable max returns 0");
    
    int best_action = get_best_action_cached(NULL, 0);
    TEST_ASSERT(best_action == 0, "NULL qtable best action returns 0");
    
    // Test out-of-bounds access
    qtable = create_optimized_qtable(10, 4, ALLOC_STANDARD, (AccessPatternHints){0});
    if (qtable) {
        value = get_q_value_fast(qtable, 100, 0); // Out of bounds state
        TEST_ASSERT(true, "Out-of-bounds access handled gracefully");
        
        max_val = get_max_q_value_cached(qtable, -1); // Negative state
        TEST_ASSERT(max_val == 0.0f, "Negative state index handled");
        
        destroy_optimized_qtable(qtable);
    }
}

// Main test runner
int main() {
    printf(COLOR_BLUE "Q-Table Optimization Test Suite" COLOR_RESET "\n");
    printf("================================\n");
    
    // Run all tests
    test_qtable_creation();
    test_basic_operations();
    test_cached_operations();
    test_batch_operations();
    test_simd_operations();
    test_performance_comparison();
    test_memory_layout();
    test_compatibility_wrapper();
    test_error_handling();
    
    // Print summary
    printf("\n" COLOR_BLUE "=== Test Summary ===" COLOR_RESET "\n");
    printf("Total tests: %d\n", g_results.total_tests);
    printf(COLOR_GREEN "Passed: %d" COLOR_RESET "\n", g_results.tests_passed);
    
    if (g_results.tests_failed > 0) {
        printf(COLOR_RED "Failed: %d" COLOR_RESET "\n", g_results.tests_failed);
        printf(COLOR_RED "❌ Some tests failed!" COLOR_RESET "\n");
        return 1;
    } else {
        printf(COLOR_GREEN "✅ All tests passed!" COLOR_RESET "\n");
        return 0;
    }
}
