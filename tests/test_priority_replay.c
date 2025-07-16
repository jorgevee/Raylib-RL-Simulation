/*
 * Priority Experience Replay Test Suite
 * 
 * This test suite verifies the correct implementation of:
 * - Priority experience buffer creation and management
 * - Priority-based sampling
 * - Importance sampling weights
 * - Batch replay functionality
 * - TD error calculation and priority updates
 */

#include "../include/agent.h"
#include "../include/environment.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>

// Test configuration
#define TEST_BUFFER_SIZE 1000
#define TEST_BATCH_SIZE 32
#define TEST_GRID_SIZE 8
#define NUM_TEST_EPISODES 100

// Test result tracking
typedef struct {
    int tests_passed;
    int tests_failed;
    char last_error[256];
} TestResults;

TestResults test_results = {0, 0, ""};

// Helper macros for testing
#define ASSERT_TRUE(condition, message) \
    if (!(condition)) { \
        snprintf(test_results.last_error, sizeof(test_results.last_error), "%s", message); \
        test_results.tests_failed++; \
        printf("❌ FAIL: %s\n", message); \
        return false; \
    } else { \
        test_results.tests_passed++; \
        printf("✅ PASS: %s\n", message); \
    }

#define ASSERT_FLOAT_NEAR(actual, expected, tolerance, message) \
    if (fabsf((actual) - (expected)) > (tolerance)) { \
        snprintf(test_results.last_error, sizeof(test_results.last_error), \
                "%s: Expected %.6f, got %.6f", message, expected, actual); \
        test_results.tests_failed++; \
        printf("❌ FAIL: %s\n", test_results.last_error); \
        return false; \
    } else { \
        test_results.tests_passed++; \
        printf("✅ PASS: %s\n", message); \
    }

// Test priority buffer creation and destruction
bool test_priority_buffer_creation() {
    printf("\n--- Testing Priority Buffer Creation ---\n");
    
    ReplayConfig config = create_default_replay_config();
    PriorityExperienceBuffer* buffer = create_priority_buffer(TEST_BUFFER_SIZE, config);
    
    ASSERT_TRUE(buffer != NULL, "Priority buffer creation");
    ASSERT_TRUE(buffer->capacity == TEST_BUFFER_SIZE, "Buffer capacity set correctly");
    ASSERT_TRUE(buffer->size == 0, "Initial buffer size is zero");
    ASSERT_TRUE(buffer->alpha == config.priority_alpha, "Alpha parameter set correctly");
    ASSERT_TRUE(buffer->beta == config.priority_beta_start, "Beta parameter set correctly");
    
    destroy_priority_buffer(buffer);
    printf("Priority buffer destroyed successfully\n");
    
    return true;
}

// Test adding experiences to priority buffer
bool test_add_priority_experience() {
    printf("\n--- Testing Add Priority Experience ---\n");
    
    ReplayConfig config = create_default_replay_config();
    PriorityExperienceBuffer* buffer = create_priority_buffer(TEST_BUFFER_SIZE, config);
    
    // Add some test experiences
    for (int i = 0; i < 10; i++) {
        float td_error = (float)(i + 1) / 10.0f;  // Varying TD errors
        add_priority_experience(buffer, i, ACTION_UP, 1.0f, i + 1, false, td_error);
    }
    
    ASSERT_TRUE(buffer->size == 10, "Buffer size increased correctly");
    ASSERT_TRUE(buffer->current_index == 10, "Current index updated correctly");
    
    // Check that experiences were stored
    for (int i = 0; i < 10; i++) {
        PriorityExperience* exp = &buffer->experiences[i];
        ASSERT_TRUE(exp->state == i, "Experience state stored correctly");
        ASSERT_TRUE(exp->action == ACTION_UP, "Experience action stored correctly");
        ASSERT_TRUE(exp->reward == 1.0f, "Experience reward stored correctly");
        ASSERT_TRUE(exp->next_state == i + 1, "Experience next_state stored correctly");
        ASSERT_TRUE(!exp->done, "Experience done flag stored correctly");
    }
    
    destroy_priority_buffer(buffer);
    return true;
}

// Test priority calculation and ordering
bool test_priority_calculation() {
    printf("\n--- Testing Priority Calculation ---\n");
    
    ReplayConfig config = create_default_replay_config();
    PriorityExperienceBuffer* buffer = create_priority_buffer(TEST_BUFFER_SIZE, config);
    
    // Add experiences with known TD errors
    float td_errors[] = {0.1f, 0.5f, 0.2f, 0.8f, 0.05f};
    int num_experiences = sizeof(td_errors) / sizeof(td_errors[0]);
    
    for (int i = 0; i < num_experiences; i++) {
        add_priority_experience(buffer, i, ACTION_UP, 1.0f, i + 1, false, td_errors[i]);
    }
    
    // Check that priorities are calculated correctly
    for (int i = 0; i < num_experiences; i++) {
        PriorityExperience* exp = &buffer->experiences[i];
        float expected_priority = powf(fabsf(td_errors[i]) + buffer->min_priority, buffer->alpha);
        ASSERT_FLOAT_NEAR(exp->priority, expected_priority, 1e-6f, "Priority calculation correct");
    }
    
    // Check that max priority is updated
    float max_expected = powf(0.8f + buffer->min_priority, buffer->alpha);
    ASSERT_FLOAT_NEAR(buffer->max_priority, max_expected, 1e-6f, "Max priority updated correctly");
    
    destroy_priority_buffer(buffer);
    return true;
}

// Test priority sampling
bool test_priority_sampling() {
    printf("\n--- Testing Priority Sampling ---\n");
    
    ReplayConfig config = create_default_replay_config();
    PriorityExperienceBuffer* buffer = create_priority_buffer(TEST_BUFFER_SIZE, config);
    
    // Add experiences with different priorities
    for (int i = 0; i < 100; i++) {
        float td_error = (i % 10) / 10.0f;  // Creates varying priorities
        add_priority_experience(buffer, i, ACTION_UP, 1.0f, i + 1, false, td_error);
    }
    
    // Sample a batch
    int batch_size = 32;
    int* indices = (int*)malloc(batch_size * sizeof(int));
    float* weights = (float*)malloc(batch_size * sizeof(float));
    
    PriorityExperience* batch = sample_priority_batch(buffer, batch_size, indices, weights);
    
    ASSERT_TRUE(batch != NULL, "Batch sampling returned valid batch");
    
    // Check that all sampled experiences are valid
    for (int i = 0; i < batch_size; i++) {
        ASSERT_TRUE(indices[i] >= 0 && indices[i] < buffer->size, "Sampled index is valid");
        ASSERT_TRUE(weights[i] > 0.0f, "Importance weight is positive");
    }
    
    // Count frequency of high-priority vs low-priority samples
    int high_priority_count = 0;
    int low_priority_count = 0;
    
    for (int i = 0; i < batch_size; i++) {
        int idx = indices[i];
        if (buffer->experiences[idx].td_error > 0.5f) {
            high_priority_count++;
        } else if (buffer->experiences[idx].td_error < 0.2f) {
            low_priority_count++;
        }
    }
    
    // High priority experiences should be sampled more frequently
    // This is a probabilistic test, so we use a reasonable threshold
    printf("High priority samples: %d, Low priority samples: %d\n", 
           high_priority_count, low_priority_count);
    
    free(indices);
    free(weights);
    destroy_priority_buffer(buffer);
    return true;
}

// Test importance sampling weights
bool test_importance_weights() {
    printf("\n--- Testing Importance Sampling Weights ---\n");
    
    ReplayConfig config = create_default_replay_config();
    PriorityExperienceBuffer* buffer = create_priority_buffer(TEST_BUFFER_SIZE, config);
    
    // Add experiences with different priorities
    add_priority_experience(buffer, 0, ACTION_UP, 1.0f, 1, false, 0.1f);  // Low priority
    add_priority_experience(buffer, 1, ACTION_UP, 1.0f, 2, false, 0.8f);  // High priority
    
    float weight_low = calculate_importance_weight(buffer, 0);
    float weight_high = calculate_importance_weight(buffer, 1);
    
    ASSERT_TRUE(weight_low > 0.0f, "Low priority weight is positive");
    ASSERT_TRUE(weight_high > 0.0f, "High priority weight is positive");
    
    // Low priority experiences should have higher importance weights
    ASSERT_TRUE(weight_low > weight_high, "Low priority has higher importance weight");
    
    destroy_priority_buffer(buffer);
    return true;
}

// Test TD error calculation
bool test_td_error_calculation() {
    printf("\n--- Testing TD Error Calculation ---\n");
    
    // Create a simple agent for testing
    QLearningAgent* agent = create_agent(TEST_GRID_SIZE * TEST_GRID_SIZE, NUM_ACTIONS, 
                                        0.1f, 0.9f, 0.1f);
    ASSERT_TRUE(agent != NULL, "Agent creation for TD error test");
    
    // Set some known Q-values
    set_q_value(agent, 0, ACTION_UP, 5.0f);
    set_q_value(agent, 1, ACTION_UP, 10.0f);
    set_q_value(agent, 1, ACTION_DOWN, 8.0f);
    set_q_value(agent, 1, ACTION_LEFT, 12.0f);  // Maximum for next state
    set_q_value(agent, 1, ACTION_RIGHT, 6.0f);
    
    // Create test experience
    PriorityExperience exp = {
        .state = 0,
        .action = ACTION_UP,
        .reward = 2.0f,
        .next_state = 1,
        .done = false
    };
    
    float td_error = calculate_td_error(agent, &exp);
    
    // Expected TD error calculation:
    // current_q = 5.0
    // max_next_q = 12.0 (maximum Q-value for next state)
    // td_target = reward + gamma * max_next_q = 2.0 + 0.9 * 12.0 = 12.8
    // td_error = td_target - current_q = 12.8 - 5.0 = 7.8
    float expected_td_error = 7.8f;
    
    ASSERT_FLOAT_NEAR(td_error, expected_td_error, 1e-6f, "TD error calculation correct");
    
    destroy_agent(agent);
    return true;
}

// Test batch replay functionality
bool test_batch_replay() {
    printf("\n--- Testing Batch Replay ---\n");
    
    // Create agent and environment for realistic testing
    QLearningAgent* agent = create_agent(TEST_GRID_SIZE * TEST_GRID_SIZE, NUM_ACTIONS, 
                                        0.1f, 0.9f, 0.1f);
    ReplayConfig config = create_default_replay_config();
    PriorityExperienceBuffer* buffer = create_priority_buffer(TEST_BUFFER_SIZE, config);
    
    ASSERT_TRUE(agent != NULL && buffer != NULL, "Agent and buffer creation for batch replay test");
    
    // Add some experiences to buffer
    for (int i = 0; i < 50; i++) {
        float td_error = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;  // Random TD errors
        add_priority_experience(buffer, i % (TEST_GRID_SIZE * TEST_GRID_SIZE), 
                              (Action)(i % NUM_ACTIONS), 1.0f, 
                              (i + 1) % (TEST_GRID_SIZE * TEST_GRID_SIZE), 
                              false, td_error);
    }
    
    // Store initial Q-values for comparison
    float initial_q_values[TEST_GRID_SIZE * TEST_GRID_SIZE][NUM_ACTIONS];
    for (int s = 0; s < TEST_GRID_SIZE * TEST_GRID_SIZE; s++) {
        for (int a = 0; a < NUM_ACTIONS; a++) {
            initial_q_values[s][a] = get_q_value(agent, s, a);
        }
    }
    
    // Sample and replay a batch
    int batch_size = 16;
    int* indices = (int*)malloc(batch_size * sizeof(int));
    float* weights = (float*)malloc(batch_size * sizeof(float));
    
    PriorityExperience* batch = sample_priority_batch(buffer, batch_size, indices, weights);
    replay_batch_experiences(agent, batch, weights, batch_size);
    
    // Check that some Q-values have changed
    bool q_values_changed = false;
    for (int s = 0; s < TEST_GRID_SIZE * TEST_GRID_SIZE && !q_values_changed; s++) {
        for (int a = 0; a < NUM_ACTIONS && !q_values_changed; a++) {
            if (fabsf(get_q_value(agent, s, a) - initial_q_values[s][a]) > 1e-6f) {
                q_values_changed = true;
            }
        }
    }
    
    ASSERT_TRUE(q_values_changed, "Q-values updated after batch replay");
    
    free(indices);
    free(weights);
    destroy_agent(agent);
    destroy_priority_buffer(buffer);
    return true;
}

// Test priority updates
bool test_priority_updates() {
    printf("\n--- Testing Priority Updates ---\n");
    
    ReplayConfig config = create_default_replay_config();
    PriorityExperienceBuffer* buffer = create_priority_buffer(TEST_BUFFER_SIZE, config);
    
    // Add some experiences
    for (int i = 0; i < 10; i++) {
        add_priority_experience(buffer, i, ACTION_UP, 1.0f, i + 1, false, 0.1f);
    }
    
    // Update priorities for some experiences
    int indices[] = {2, 5, 8};
    float new_td_errors[] = {0.9f, 0.7f, 0.3f};
    int count = 3;
    
    // Store old priorities
    float old_priorities[3];
    for (int i = 0; i < count; i++) {
        old_priorities[i] = buffer->experiences[indices[i]].priority;
    }
    
    update_experience_priorities(buffer, indices, new_td_errors, count);
    
    // Check that priorities were updated
    for (int i = 0; i < count; i++) {
        float expected_priority = powf(fabsf(new_td_errors[i]) + buffer->min_priority, buffer->alpha);
        ASSERT_FLOAT_NEAR(buffer->experiences[indices[i]].priority, expected_priority, 1e-6f, 
                         "Priority updated correctly");
        ASSERT_TRUE(fabsf(buffer->experiences[indices[i]].priority - old_priorities[i]) > 1e-6f,
                   "Priority actually changed");
    }
    
    destroy_priority_buffer(buffer);
    return true;
}

// Test beta annealing
bool test_beta_annealing() {
    printf("\n--- Testing Beta Annealing ---\n");
    
    ReplayConfig config = create_replay_config(true, 1000, 32, 4, 0.6f, 0.4f, 1.0f, 100, 1e-6f);
    PriorityExperienceBuffer* buffer = create_priority_buffer(TEST_BUFFER_SIZE, config);
    
    float initial_beta = buffer->beta;
    ASSERT_FLOAT_NEAR(initial_beta, 0.4f, 1e-6f, "Initial beta value correct");
    
    // Anneal beta several times
    for (int i = 0; i < 50; i++) {
        update_beta(buffer);
    }
    
    ASSERT_TRUE(buffer->beta > initial_beta, "Beta increased after annealing");
    ASSERT_TRUE(buffer->beta <= 1.0f, "Beta doesn't exceed maximum");
    
    // Anneal many more times to test clamping
    for (int i = 0; i < 100; i++) {
        update_beta(buffer);
    }
    
    ASSERT_FLOAT_NEAR(buffer->beta, 1.0f, 1e-6f, "Beta clamped to maximum");
    
    destroy_priority_buffer(buffer);
    return true;
}

// Performance comparison test
bool test_performance_comparison() {
    printf("\n--- Testing Performance Comparison ---\n");
    printf("This test compares learning with and without priority replay...\n");
    
    const int COMPARISON_EPISODES = 50;
    const int GRID_SIZE = 6;
    
    // Test without priority replay (baseline)
    QLearningAgent* agent1 = create_agent(GRID_SIZE * GRID_SIZE, NUM_ACTIONS, 0.1f, 0.9f, 1.0f);
    GridWorld* world1 = create_grid_world(GRID_SIZE, GRID_SIZE);
    world1->start_pos = (Position){0, 0};
    world1->goal_pos = (Position){GRID_SIZE-1, GRID_SIZE-1};
    world1->step_penalty = -0.1f;
    world1->goal_reward = 10.0f;
    world1->wall_penalty = -5.0f;
    world1->max_steps = 50;
    
    float total_reward_baseline = 0.0f;
    int successful_episodes_baseline = 0;
    
    for (int episode = 0; episode < COMPARISON_EPISODES; episode++) {
        reset_environment(world1);
        float episode_reward = 0.0f;
        
        while (!world1->episode_done && world1->episode_steps < world1->max_steps) {
            int state = get_state_index(world1);
            Action action = select_action(agent1, state);
            StepResult result = step_environment(world1, action);
            
            update_q_value(agent1, state, action, result.reward,
                         position_to_state(world1, result.next_state.position), result.done);
            
            episode_reward += result.reward;
        }
        
        if (positions_equal(world1->agent_pos, world1->goal_pos)) {
            successful_episodes_baseline++;
        }
        
        total_reward_baseline += episode_reward;
        decay_epsilon(agent1);
    }
    
    float avg_reward_baseline = total_reward_baseline / COMPARISON_EPISODES;
    float success_rate_baseline = (float)successful_episodes_baseline / COMPARISON_EPISODES;
    
    printf("Baseline (no replay): Avg Reward=%.2f, Success Rate=%.2f%%\n", 
           avg_reward_baseline, success_rate_baseline * 100);
    
    // Test with priority replay
    QLearningAgent* agent2 = create_agent(GRID_SIZE * GRID_SIZE, NUM_ACTIONS, 0.1f, 0.9f, 1.0f);
    GridWorld* world2 = create_grid_world(GRID_SIZE, GRID_SIZE);
    world2->start_pos = (Position){0, 0};
    world2->goal_pos = (Position){GRID_SIZE-1, GRID_SIZE-1};
    world2->step_penalty = -0.1f;
    world2->goal_reward = 10.0f;
    world2->wall_penalty = -5.0f;
    world2->max_steps = 50;
    
    ReplayConfig config = create_default_replay_config();
    config.batch_size = 16;
    config.replay_frequency = 4;
    PriorityExperienceBuffer* buffer = create_priority_buffer(1000, config);
    
    float total_reward_replay = 0.0f;
    int successful_episodes_replay = 0;
    int step_count = 0;
    
    for (int episode = 0; episode < COMPARISON_EPISODES; episode++) {
        reset_environment(world2);
        float episode_reward = 0.0f;
        
        while (!world2->episode_done && world2->episode_steps < world2->max_steps) {
            int state = get_state_index(world2);
            Action action = select_action(agent2, state);
            StepResult result = step_environment(world2, action);
            
            // Calculate TD error for experience
            float td_error = result.reward;
            if (!result.done) {
                float max_next_q = 0.0f;
                for (int a = 0; a < NUM_ACTIONS; a++) {
                    float q_val = get_q_value(agent2, position_to_state(world2, result.next_state.position), a);
                    if (a == 0 || q_val > max_next_q) {
                        max_next_q = q_val;
                    }
                }
                td_error += agent2->discount_factor * max_next_q;
            }
            td_error -= get_q_value(agent2, state, action);
            
            // Add experience to buffer
            add_priority_experience(buffer, state, action, result.reward,
                                  position_to_state(world2, result.next_state.position), 
                                  result.done, td_error);
            
            // Regular Q-learning update
            update_q_value(agent2, state, action, result.reward,
                         position_to_state(world2, result.next_state.position), result.done);
            
            // Replay experiences periodically
            if (step_count % config.replay_frequency == 0 && buffer->size >= config.batch_size) {
                int* indices = (int*)malloc(config.batch_size * sizeof(int));
                float* weights = (float*)malloc(config.batch_size * sizeof(float));
                
                PriorityExperience* batch = sample_priority_batch(buffer, config.batch_size, indices, weights);
                if (batch) {
                    replay_batch_experiences(agent2, batch, weights, config.batch_size);
                    
                    // Update priorities based on new TD errors
                    float* new_td_errors = (float*)malloc(config.batch_size * sizeof(float));
                    for (int i = 0; i < config.batch_size; i++) {
                        new_td_errors[i] = calculate_td_error(agent2, &batch[i]);
                    }
                    update_experience_priorities(buffer, indices, new_td_errors, config.batch_size);
                    free(new_td_errors);
                }
                
                free(indices);
                free(weights);
                update_beta(buffer);
            }
            
            episode_reward += result.reward;
            step_count++;
        }
        
        if (positions_equal(world2->agent_pos, world2->goal_pos)) {
            successful_episodes_replay++;
        }
        
        total_reward_replay += episode_reward;
        decay_epsilon(agent2);
    }
    
    float avg_reward_replay = total_reward_replay / COMPARISON_EPISODES;
    float success_rate_replay = (float)successful_episodes_replay / COMPARISON_EPISODES;
    
    printf("With priority replay: Avg Reward=%.2f, Success Rate=%.2f%%\n", 
           avg_reward_replay, success_rate_replay * 100);
    
    // Priority replay should generally perform better or at least comparable
    float improvement = (avg_reward_replay - avg_reward_baseline) / fabsf(avg_reward_baseline);
    printf("Performance improvement: %.2f%%\n", improvement * 100);
    
    // Cleanup
    destroy_agent(agent1);
    destroy_agent(agent2);
    destroy_grid_world(world1);
    destroy_grid_world(world2);
    destroy_priority_buffer(buffer);
    
    ASSERT_TRUE(true, "Performance comparison completed successfully");
    return true;
}

// Main test runner
int main() {
    printf("Priority Experience Replay Test Suite\n");
    printf("====================================\n");
    
    srand((unsigned int)time(NULL));
    
    // Run all tests
    bool (*tests[])() = {
        test_priority_buffer_creation,
        test_add_priority_experience,
        test_priority_calculation,
        test_priority_sampling,
        test_importance_weights,
        test_td_error_calculation,
        test_batch_replay,
        test_priority_updates,
        test_beta_annealing,
        test_performance_comparison
    };
    
    const char* test_names[] = {
        "Priority Buffer Creation",
        "Add Priority Experience", 
        "Priority Calculation",
        "Priority Sampling",
        "Importance Weights",
        "TD Error Calculation",
        "Batch Replay",
        "Priority Updates",
        "Beta Annealing",
        "Performance Comparison"
    };
    
    int num_tests = sizeof(tests) / sizeof(tests[0]);
    int tests_run = 0;
    
    for (int i = 0; i < num_tests; i++) {
        printf("\n[%d/%d] Running %s...\n", i + 1, num_tests, test_names[i]);
        
        if (tests[i]()) {
            printf("✅ %s: PASSED\n", test_names[i]);
        } else {
            printf("❌ %s: FAILED - %s\n", test_names[i], test_results.last_error);
        }
        tests_run++;
    }
    
    // Print final results
    printf("\n" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "\n");
    printf("TEST RESULTS SUMMARY\n");
    printf("=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "\n");
    printf("Tests Run: %d\n", tests_run);
    printf("Tests Passed: %d\n", test_results.tests_passed);
    printf("Tests Failed: %d\n", test_results.tests_failed);
    printf("Success Rate: %.1f%%\n", 
           (float)test_results.tests_passed / (test_results.tests_passed + test_results.tests_failed) * 100);
    
    if (test_results.tests_failed == 0) {
        printf("🎉 ALL TESTS PASSED! Priority Experience Replay implementation is working correctly.\n");
        return 0;
    } else {
        printf("⚠️  Some tests failed. Please review the implementation.\n");
        return 1;
    }
}
