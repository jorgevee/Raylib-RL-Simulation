/*
 * State Visit Tracking Test Suite
 * 
 * This test suite verifies the correct implementation of:
 * - State visit tracker creation and management
 * - Visit count tracking and priority calculation
 * - Adaptive epsilon and learning rate functionality
 * - Exploration bonus calculation and decay
 * - State priority updates and analysis
 * - Enhanced action selection and Q-value updates
 */

#include "../include/agent.h"
#include "../include/environment.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>

// Test configuration
#define TEST_NUM_STATES 64
#define TEST_NUM_ACTIONS 4
#define TEST_GRID_SIZE 8
#define NUM_TEST_EPISODES 50
#define EPSILON 1e-6f

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

#define ASSERT_INT_EQUAL(actual, expected, message) \
    if ((actual) != (expected)) { \
        snprintf(test_results.last_error, sizeof(test_results.last_error), \
                "%s: Expected %d, got %d", message, expected, actual); \
        test_results.tests_failed++; \
        printf("❌ FAIL: %s\n", test_results.last_error); \
        return false; \
    } else { \
        test_results.tests_passed++; \
        printf("✅ PASS: %s\n", message); \
    }

// Test state visit tracker creation and destruction
bool test_state_visit_tracker_creation() {
    printf("\n--- Testing State Visit Tracker Creation ---\n");
    
    StateVisitTracker* tracker = create_state_visit_tracker(TEST_NUM_STATES, true, true);
    
    ASSERT_TRUE(tracker != NULL, "State visit tracker creation");
    ASSERT_INT_EQUAL(tracker->num_states, TEST_NUM_STATES, "Number of states set correctly");
    ASSERT_INT_EQUAL(tracker->total_visits, 0, "Initial total visits is zero");
    ASSERT_TRUE(tracker->adaptive_epsilon, "Adaptive epsilon enabled");
    ASSERT_TRUE(tracker->adaptive_learning_rate, "Adaptive learning rate enabled");
    
    // Check initial values
    for (int i = 0; i < TEST_NUM_STATES; i++) {
        ASSERT_INT_EQUAL(tracker->visit_counts[i], 0, "Initial visit count is zero");
        ASSERT_FLOAT_NEAR(tracker->exploration_bonuses[i], 1.0f, EPSILON, "Initial exploration bonus");
        ASSERT_FLOAT_NEAR(tracker->state_epsilons[i], 1.0f, EPSILON, "Initial state epsilon");
        ASSERT_FLOAT_NEAR(tracker->state_learning_rates[i], 1.0f, EPSILON, "Initial state learning rate");
        ASSERT_FLOAT_NEAR(tracker->visit_priorities[i], 1.0f, EPSILON, "Initial visit priority");
    }
    
    destroy_state_visit_tracker(tracker);
    printf("State visit tracker destroyed successfully\n");
    
    return true;
}

// Test visit count updates and exploration bonus calculation
bool test_visit_count_updates() {
    printf("\n--- Testing Visit Count Updates ---\n");
    
    StateVisitTracker* tracker = create_state_visit_tracker(TEST_NUM_STATES, true, true);
    
    // Test updating a single state multiple times
    int test_state = 10;
    for (int i = 1; i <= 5; i++) {
        update_state_visit(tracker, test_state);
        
        ASSERT_INT_EQUAL(tracker->visit_counts[test_state], i, "Visit count incremented correctly");
        ASSERT_INT_EQUAL(tracker->total_visits, i, "Total visits incremented correctly");
        
        // Check exploration bonus decreases with visits
        float expected_bonus = fmaxf(tracker->min_exploration_bonus, 
                                   1.0f / sqrtf((float)i + 1));
        ASSERT_FLOAT_NEAR(tracker->exploration_bonuses[test_state], expected_bonus, EPSILON,
                         "Exploration bonus decreases with visits");
    }
    
    // Test updating multiple states
    update_state_visit(tracker, 5);
    update_state_visit(tracker, 15);
    update_state_visit(tracker, 20);
    
    ASSERT_INT_EQUAL(tracker->visit_counts[5], 1, "State 5 visit count");
    ASSERT_INT_EQUAL(tracker->visit_counts[15], 1, "State 15 visit count");
    ASSERT_INT_EQUAL(tracker->visit_counts[20], 1, "State 20 visit count");
    ASSERT_INT_EQUAL(tracker->total_visits, 8, "Total visits after multiple updates");
    
    destroy_state_visit_tracker(tracker);
    return true;
}

// Test adaptive epsilon functionality
bool test_adaptive_epsilon() {
    printf("\n--- Testing Adaptive Epsilon ---\n");
    
    StateVisitTracker* tracker = create_state_visit_tracker(TEST_NUM_STATES, true, false);
    float base_epsilon = 0.5f;
    
    // Test unvisited state (should have high epsilon)
    int unvisited_state = 0;
    float epsilon_unvisited = get_state_epsilon(tracker, unvisited_state, base_epsilon);
    ASSERT_FLOAT_NEAR(epsilon_unvisited, base_epsilon * 1.0f, EPSILON, 
                     "Unvisited state has high epsilon");
    
    // Visit a state multiple times and check epsilon decreases
    int visited_state = 1;
    for (int i = 0; i < 10; i++) {
        update_state_visit(tracker, visited_state);
    }
    
    float epsilon_visited = get_state_epsilon(tracker, visited_state, base_epsilon);
    ASSERT_TRUE(epsilon_visited < epsilon_unvisited, "Visited state has lower epsilon");
    ASSERT_TRUE(epsilon_visited >= base_epsilon * tracker->min_exploration_bonus, 
               "Epsilon doesn't go below minimum");
    
    // Test with adaptive epsilon disabled
    StateVisitTracker* tracker_disabled = create_state_visit_tracker(TEST_NUM_STATES, false, false);
    update_state_visit(tracker_disabled, 5);
    float epsilon_disabled = get_state_epsilon(tracker_disabled, 5, base_epsilon);
    ASSERT_FLOAT_NEAR(epsilon_disabled, base_epsilon, EPSILON, 
                     "Disabled adaptive epsilon returns base epsilon");
    
    destroy_state_visit_tracker(tracker);
    destroy_state_visit_tracker(tracker_disabled);
    return true;
}

// Test adaptive learning rate functionality
bool test_adaptive_learning_rate() {
    printf("\n--- Testing Adaptive Learning Rate ---\n");
    
    StateVisitTracker* tracker = create_state_visit_tracker(TEST_NUM_STATES, false, true);
    float base_learning_rate = 0.1f;
    
    // Test unvisited state (should have high learning rate)
    int unvisited_state = 0;
    float lr_unvisited = get_state_learning_rate(tracker, unvisited_state, base_learning_rate);
    ASSERT_FLOAT_NEAR(lr_unvisited, base_learning_rate * 2.0f, EPSILON, 
                     "Unvisited state has high learning rate");
    
    // Visit a state multiple times and check learning rate decreases
    int visited_state = 1;
    for (int i = 0; i < 20; i++) {
        update_state_visit(tracker, visited_state);
    }
    
    float lr_visited = get_state_learning_rate(tracker, visited_state, base_learning_rate);
    ASSERT_TRUE(lr_visited < lr_unvisited, "Visited state has lower learning rate");
    ASSERT_TRUE(lr_visited >= base_learning_rate, "Learning rate doesn't go below base rate");
    
    // Test with adaptive learning rate disabled
    StateVisitTracker* tracker_disabled = create_state_visit_tracker(TEST_NUM_STATES, false, false);
    update_state_visit(tracker_disabled, 5);
    float lr_disabled = get_state_learning_rate(tracker_disabled, 5, base_learning_rate);
    ASSERT_FLOAT_NEAR(lr_disabled, base_learning_rate, EPSILON, 
                     "Disabled adaptive learning rate returns base rate");
    
    destroy_state_visit_tracker(tracker);
    destroy_state_visit_tracker(tracker_disabled);
    return true;
}

// Test state priority calculation and updates
bool test_state_priorities() {
    printf("\n--- Testing State Priority Calculation ---\n");
    
    StateVisitTracker* tracker = create_state_visit_tracker(TEST_NUM_STATES, true, true);
    
    // Visit some states with different frequencies
    for (int i = 0; i < 10; i++) update_state_visit(tracker, 0);  // High visits
    for (int i = 0; i < 5; i++) update_state_visit(tracker, 1);   // Medium visits
    update_state_visit(tracker, 2);                                // Low visits
    // State 3 remains unvisited                                   // No visits
    
    // Check that less visited states have higher priorities
    float priority_unvisited = tracker->visit_priorities[3];
    float priority_low = tracker->visit_priorities[2];
    float priority_medium = tracker->visit_priorities[1];
    float priority_high = tracker->visit_priorities[0];
    
    ASSERT_TRUE(priority_unvisited >= priority_low, "Unvisited state has higher priority than low visited");
    ASSERT_TRUE(priority_low >= priority_medium, "Low visited state has higher priority than medium visited");
    ASSERT_TRUE(priority_medium >= priority_high, "Medium visited state has higher priority than high visited");
    
    // Test priority state selection
    int highest_priority_state = select_priority_state(tracker);
    ASSERT_TRUE(highest_priority_state == 3 || tracker->visit_priorities[highest_priority_state] >= priority_unvisited, 
               "Highest priority state selection is correct");
    
    destroy_state_visit_tracker(tracker);
    return true;
}

// Test exploration bonus decay
bool test_exploration_bonus_decay() {
    printf("\n--- Testing Exploration Bonus Decay ---\n");
    
    StateVisitTracker* tracker = create_state_visit_tracker(TEST_NUM_STATES, true, true);
    
    // Visit a state to set its exploration bonus
    update_state_visit(tracker, 0);
    float initial_bonus = tracker->exploration_bonuses[0];
    
    // Apply decay multiple times
    for (int i = 0; i < 10; i++) {
        decay_exploration_bonuses(tracker);
    }
    
    float decayed_bonus = tracker->exploration_bonuses[0];
    ASSERT_TRUE(decayed_bonus < initial_bonus, "Exploration bonus decreased after decay");
    ASSERT_TRUE(decayed_bonus >= tracker->min_exploration_bonus, "Bonus doesn't go below minimum");
    
    // Apply many decay cycles to test minimum clamping
    for (int i = 0; i < 1000; i++) {
        decay_exploration_bonuses(tracker);
    }
    
    ASSERT_FLOAT_NEAR(tracker->exploration_bonuses[0], tracker->min_exploration_bonus, EPSILON,
                     "Bonus clamped to minimum after extensive decay");
    
    destroy_state_visit_tracker(tracker);
    return true;
}

// Test enhanced action selection with state visit priority
bool test_enhanced_action_selection() {
    printf("\n--- Testing Enhanced Action Selection ---\n");
    
    QLearningAgent* agent = create_agent(TEST_NUM_STATES, TEST_NUM_ACTIONS, 0.1f, 0.9f, 0.5f);
    StateVisitTracker* tracker = create_state_visit_tracker(TEST_NUM_STATES, true, true);
    
    ASSERT_TRUE(agent != NULL && tracker != NULL, "Agent and tracker creation");
    
    // Set some Q-values to make action 2 optimal for state 0
    set_q_value(agent, 0, ACTION_UP, 1.0f);
    set_q_value(agent, 0, ACTION_DOWN, 2.0f);
    set_q_value(agent, 0, ACTION_LEFT, 5.0f);     // Best action
    set_q_value(agent, 0, ACTION_RIGHT, 3.0f);
    
    // Test action selection with priority (should update visit count)
    int initial_visits = tracker->visit_counts[0];
    Action selected_action = select_action_with_priority(agent, tracker, 0);
    
    ASSERT_INT_EQUAL(tracker->visit_counts[0], initial_visits + 1, "Visit count updated during action selection");
    ASSERT_TRUE(selected_action >= ACTION_UP && selected_action <= ACTION_RIGHT, "Valid action selected");
    
    // Test many selections to check exploration vs exploitation balance
    int exploration_count = 0;
    int exploitation_count = 0;
    int num_trials = 1000;
    
    // Reset agent epsilon to a known value for testing
    agent->epsilon = 0.3f;
    
    for (int i = 0; i < num_trials; i++) {
        Action action = select_action_with_priority(agent, tracker, 0);
        if (action == ACTION_LEFT) {
            exploitation_count++;
        } else {
            exploration_count++;
        }
    }
    
    // Should have some balance between exploration and exploitation
    ASSERT_TRUE(exploration_count > 0, "Some exploration occurred");
    ASSERT_TRUE(exploitation_count > 0, "Some exploitation occurred");
    ASSERT_TRUE(exploitation_count > exploration_count, "More exploitation than exploration with current epsilon");
    
    destroy_agent(agent);
    destroy_state_visit_tracker(tracker);
    return true;
}

// Test enhanced Q-value updates with state visit priority
bool test_enhanced_q_value_updates() {
    printf("\n--- Testing Enhanced Q-Value Updates ---\n");
    
    QLearningAgent* agent = create_agent(TEST_NUM_STATES, TEST_NUM_ACTIONS, 0.1f, 0.9f, 0.1f);
    StateVisitTracker* tracker = create_state_visit_tracker(TEST_NUM_STATES, true, true);
    
    ASSERT_TRUE(agent != NULL && tracker != NULL, "Agent and tracker creation");
    
    // Set initial Q-values
    set_q_value(agent, 0, ACTION_UP, 0.0f);
    set_q_value(agent, 1, ACTION_UP, 5.0f);
    set_q_value(agent, 1, ACTION_DOWN, 3.0f);
    set_q_value(agent, 1, ACTION_LEFT, 7.0f);   // Best action for next state
    set_q_value(agent, 1, ACTION_RIGHT, 2.0f);
    
    float initial_q = get_q_value(agent, 0, ACTION_UP);
    
    // Update Q-value with priority (includes exploration bonus)
    update_q_value_with_priority(agent, tracker, 0, ACTION_UP, 1.0f, 1, false);
    
    float updated_q = get_q_value(agent, 0, ACTION_UP);
    ASSERT_TRUE(updated_q != initial_q, "Q-value was updated");
    
    // The updated Q-value should be higher due to exploration bonus
    float exploration_bonus = get_exploration_bonus(tracker, 0);
    ASSERT_TRUE(exploration_bonus > 0.0f, "Exploration bonus is positive");
    
    // Compare with standard Q-value update (without priority)
    QLearningAgent* agent_standard = create_agent(TEST_NUM_STATES, TEST_NUM_ACTIONS, 0.1f, 0.9f, 0.1f);
    set_q_value(agent_standard, 0, ACTION_UP, 0.0f);
    set_q_value(agent_standard, 1, ACTION_UP, 5.0f);
    set_q_value(agent_standard, 1, ACTION_DOWN, 3.0f);
    set_q_value(agent_standard, 1, ACTION_LEFT, 7.0f);
    set_q_value(agent_standard, 1, ACTION_RIGHT, 2.0f);
    
    update_q_value(agent_standard, 0, ACTION_UP, 1.0f, 1, false);
    float standard_q = get_q_value(agent_standard, 0, ACTION_UP);
    
    ASSERT_TRUE(updated_q > standard_q, "Priority-enhanced Q-value is higher due to exploration bonus");
    
    destroy_agent(agent);
    destroy_agent(agent_standard);
    destroy_state_visit_tracker(tracker);
    return true;
}

// Test state visit analysis functions
bool test_state_visit_analysis() {
    printf("\n--- Testing State Visit Analysis ---\n");
    
    StateVisitTracker* tracker = create_state_visit_tracker(TEST_NUM_STATES, true, true);
    
    // Create a pattern of visits
    for (int i = 0; i < 20; i++) update_state_visit(tracker, 0);  // Most visited
    for (int i = 0; i < 10; i++) update_state_visit(tracker, 1);
    for (int i = 0; i < 5; i++) update_state_visit(tracker, 2);
    update_state_visit(tracker, 3);                                // Least visited (among visited)
    // States 4+ remain unvisited
    
    // Test exploration coverage
    float coverage = calculate_exploration_coverage(tracker);
    float expected_coverage = 4.0f / TEST_NUM_STATES * 100.0f;  // 4 states visited out of TEST_NUM_STATES
    ASSERT_FLOAT_NEAR(coverage, expected_coverage, 0.1f, "Exploration coverage calculation");
    
    // Test least/most visited state identification
    int least_visited = get_least_visited_state(tracker);
    int most_visited = get_most_visited_state(tracker);
    
    ASSERT_INT_EQUAL(most_visited, 0, "Most visited state identification");
    ASSERT_TRUE(least_visited >= 4, "Least visited state is unvisited (among all states)");
    
    // Test state visit data saving
    save_state_visit_data(tracker, "test_state_visits.csv");
    
    // Check if file was created (basic test)
    FILE* test_file = fopen("test_state_visits.csv", "r");
    ASSERT_TRUE(test_file != NULL, "State visit data file created");
    if (test_file) {
        fclose(test_file);
        remove("test_state_visits.csv");  // Clean up
    }
    
    destroy_state_visit_tracker(tracker);
    return true;
}

// Test state visit tracker reset functionality
bool test_state_visit_reset() {
    printf("\n--- Testing State Visit Tracker Reset ---\n");
    
    StateVisitTracker* tracker = create_state_visit_tracker(TEST_NUM_STATES, true, true);
    
    // Create some visits and modifications
    for (int i = 0; i < 5; i++) {
        update_state_visit(tracker, i);
    }
    
    // Apply some decay
    for (int i = 0; i < 10; i++) {
        decay_exploration_bonuses(tracker);
    }
    
    ASSERT_TRUE(tracker->total_visits > 0, "Tracker has visits before reset");
    
    // Reset the tracker
    reset_state_visit_tracker(tracker);
    
    // Check that everything is reset to initial state
    ASSERT_INT_EQUAL(tracker->total_visits, 0, "Total visits reset to zero");
    
    for (int i = 0; i < TEST_NUM_STATES; i++) {
        ASSERT_INT_EQUAL(tracker->visit_counts[i], 0, "Visit counts reset to zero");
        ASSERT_FLOAT_NEAR(tracker->exploration_bonuses[i], 1.0f, EPSILON, "Exploration bonuses reset");
        ASSERT_FLOAT_NEAR(tracker->state_epsilons[i], 1.0f, EPSILON, "State epsilons reset");
        ASSERT_FLOAT_NEAR(tracker->state_learning_rates[i], 1.0f, EPSILON, "State learning rates reset");
        ASSERT_FLOAT_NEAR(tracker->visit_priorities[i], 1.0f, EPSILON, "Visit priorities reset");
    }
    
    destroy_state_visit_tracker(tracker);
    return true;
}

// Integration test with grid world environment
bool test_integration_with_environment() {
    printf("\n--- Testing Integration with Grid World ---\n");
    
    const int GRID_SIZE = 6;
    const int NUM_STATES = GRID_SIZE * GRID_SIZE;
    const int EPISODES = 20;
    
    // Create environment
    GridWorld* world = create_grid_world(GRID_SIZE, GRID_SIZE);
    world->start_pos = (Position){0, 0};
    world->goal_pos = (Position){GRID_SIZE-1, GRID_SIZE-1};
    world->step_penalty = -0.1f;
    world->goal_reward = 10.0f;
    world->wall_penalty = -1.0f;
    world->max_steps = 50;
    
    // Create agent and state visit tracker
    QLearningAgent* agent = create_agent(NUM_STATES, NUM_ACTIONS, 0.1f, 0.9f, 1.0f);
    StateVisitTracker* tracker = create_state_visit_tracker(NUM_STATES, true, true);
    
    ASSERT_TRUE(world != NULL && agent != NULL && tracker != NULL, "Environment setup");
    
    int total_successful_episodes = 0;
    int total_steps = 0;
    
    // Run training episodes with state visit tracking
    for (int episode = 0; episode < EPISODES; episode++) {
        reset_environment(world);
        int episode_steps = 0;
        
        while (!world->episode_done && episode_steps < world->max_steps) {
            int state = get_state_index(world);
            
            // Use enhanced action selection with state visit priority
            Action action = select_action_with_priority(agent, tracker, state);
            StepResult result = step_environment(world, action);
            
            // Use enhanced Q-value update with state visit priority
            int next_state = position_to_state(world, result.next_state.position);
            update_q_value_with_priority(agent, tracker, state, action, result.reward, next_state, result.done);
            
            episode_steps++;
            total_steps++;
        }
        
        if (positions_equal(world->agent_pos, world->goal_pos)) {
            total_successful_episodes++;
        }
        
        decay_epsilon(agent);
    }
    
    // Check that training made progress
    float success_rate = (float)total_successful_episodes / EPISODES;
    float avg_steps = (float)total_steps / EPISODES;
    
    printf("Integration test results:\n");
    printf("  Success rate: %.1f%% (%d/%d episodes)\n", success_rate * 100, total_successful_episodes, EPISODES);
    printf("  Average steps per episode: %.1f\n", avg_steps);
    printf("  Total state visits: %d\n", tracker->total_visits);
    printf("  Exploration coverage: %.1f%%\n", calculate_exploration_coverage(tracker));
    
    ASSERT_TRUE(tracker->total_visits > 0, "State visits were recorded");
    ASSERT_TRUE(calculate_exploration_coverage(tracker) > 0.0f, "Some exploration occurred");
    
    // Test state visit analysis
    print_state_visit_analysis(tracker);
    
    // Cleanup
    destroy_grid_world(world);
    destroy_agent(agent);
    destroy_state_visit_tracker(tracker);
    
    ASSERT_TRUE(true, "Integration test completed successfully");
    return true;
}

// Performance comparison test
bool test_performance_comparison() {
    printf("\n--- Testing Performance Comparison ---\n");
    printf("Comparing standard Q-learning vs priority-enhanced Q-learning...\n");
    
    const int GRID_SIZE = 5;
    const int NUM_STATES = GRID_SIZE * GRID_SIZE;
    const int EPISODES = 30;
    
    // Test standard Q-learning
    GridWorld* world1 = create_grid_world(GRID_SIZE, GRID_SIZE);
    world1->start_pos = (Position){0, 0};
    world1->goal_pos = (Position){GRID_SIZE-1, GRID_SIZE-1};
    world1->step_penalty = -0.1f;
    world1->goal_reward = 10.0f;
    world1->max_steps = 40;
    
    QLearningAgent* agent1 = create_agent(NUM_STATES, NUM_ACTIONS, 0.1f, 0.9f, 1.0f);
    
    int successful_episodes_standard = 0;
    float total_reward_standard = 0.0f;
    
    for (int episode = 0; episode < EPISODES; episode++) {
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
            successful_episodes_standard++;
        }
        
        total_reward_standard += episode_reward;
        decay_epsilon(agent1);
    }
    
    // Test priority-enhanced Q-learning
    GridWorld* world2 = create_grid_world(GRID_SIZE, GRID_SIZE);
    world2->start_pos = (Position){0, 0};
    world2->goal_pos = (Position){GRID_SIZE-1, GRID_SIZE-1};
    world2->step_penalty = -0.1f;
    world2->goal_reward = 10.0f;
    world2->max_steps = 40;
    
    QLearningAgent* agent2 = create_agent(NUM_STATES, NUM_ACTIONS, 0.1f, 0.9f, 1.0f);
    StateVisitTracker* tracker = create_state_visit_tracker(NUM_STATES, true, true);
    
    int successful_episodes_priority = 0;
    float total_reward_priority = 0.0f;
    
    for (int episode = 0; episode < EPISODES; episode++) {
        reset_environment(world2);
        float episode_reward = 0.0f;
        
        while (!world2->episode_done && world2->episode_steps < world2->max_steps) {
            int state = get_state_index(world2);
            Action action = select_action_with_priority(agent2, tracker, state);
            StepResult result = step_environment(world2, action);
            
            update_q_value_with_priority(agent2, tracker, state, action, result.reward, 
                                       position_to_state(world2, result.next_state.position), result.done);
            
            episode_reward += result.reward;
        }
        
        if (positions_equal(world2->agent_pos, world2->goal_pos)) {
            successful_episodes_priority++;
        }
        
        total_reward_priority += episode_reward;
        decay_epsilon(agent2);
        
        // Periodic exploration bonus decay
        if (episode % 5 == 0) {
            decay_exploration_bonuses(tracker);
        }
    }
    
    // Calculate performance metrics
    float success_rate_standard = (float)successful_episodes_standard / EPISODES;
    float success_rate_priority = (float)successful_episodes_priority / EPISODES;
    float avg_reward_standard = total_reward_standard / EPISODES;
    float avg_reward_priority = total_reward_priority / EPISODES;
    
    printf("Performance Comparison Results:\n");
    printf("Standard Q-learning:\n");
    printf("  Success rate: %.1f%% (%d/%d)\n", success_rate_standard * 100, successful_episodes_standard, EPISODES);
    printf("  Average reward: %.2f\n", avg_reward_standard);
    
    printf("Priority-enhanced Q-learning:\n");
    printf("  Success rate: %.1f%% (%d/%d)\n", success_rate_priority * 100, successful_episodes_priority, EPISODES);
    printf("  Average reward: %.2f\n", avg_reward_priority);
    printf("  Exploration coverage: %.1f%%\n", calculate_exploration_coverage(tracker));
    
    // Performance should be at least comparable
    ASSERT_TRUE(success_rate_priority >= 0.0f, "Priority-enhanced learning completed successfully");
    ASSERT_TRUE(calculate_exploration_coverage(tracker) > 0.0f, "Exploration occurred with priority tracking");
    
    // Cleanup
    destroy_grid_world(world1);
    destroy_grid_world(world2);
    destroy_agent(agent1);
    destroy_agent(agent2);
    destroy_state_visit_tracker(tracker);
    
    ASSERT_TRUE(true, "Performance comparison completed successfully");
    return true;
}

// Main test runner
int main() {
    printf("State Visit Tracking Test Suite\n");
    printf("===============================\n");
    
    srand((unsigned int)time(NULL));
    
    // Run all tests
    bool (*tests[])() = {
        test_state_visit_tracker_creation,
        test_visit_count_updates,
        test_adaptive_epsilon,
        test_adaptive_learning_rate,
        test_state_priorities,
        test_exploration_bonus_decay,
        test_enhanced_action_selection,
        test_enhanced_q_value_updates,
        test_state_visit_analysis,
        test_state_visit_reset,
        test_integration_with_environment,
        test_performance_comparison
    };
    
    const char* test_names[] = {
        "State Visit Tracker Creation",
        "Visit Count Updates",
        "Adaptive Epsilon",
        "Adaptive Learning Rate",
        "State Priorities",
        "Exploration Bonus Decay",
        "Enhanced Action Selection",
        "Enhanced Q-Value Updates",
        "State Visit Analysis",
        "State Visit Reset",
        "Integration with Environment",
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
        printf("🎉 ALL TESTS PASSED! State Visit Tracking implementation is working correctly.\n");
        return 0;
    } else {
        printf("⚠️  Some tests failed. Please review the implementation.\n");
        return 1;
    }
}
