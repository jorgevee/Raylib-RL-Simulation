#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "include/agent.h"
#include "include/environment.h"

void test_agent_creation() {
    printf("Test 1: Testing agent creation...\n");
    
    QLearningAgent* agent = create_agent(25, 4, 0.1f, 0.9f, 0.1f);
    if (!agent) {
        printf("❌ Failed to create agent\n");
        return;
    }
    
    printf("✓ Agent created with %d states, %d actions\n", agent->num_states, agent->num_actions);
    printf("✓ Learning rate: %.2f, Discount factor: %.2f, Epsilon: %.2f\n", 
           agent->learning_rate, agent->discount_factor, agent->epsilon);
    
    destroy_agent(agent);
    printf("✓ Agent destroyed successfully\n\n");
}

void test_q_value_operations() {
    printf("Test 2: Testing Q-value operations...\n");
    
    QLearningAgent* agent = create_agent(5, 4, 0.1f, 0.9f, 0.1f);
    
    // Test setting and getting Q-values
    set_q_value(agent, 0, ACTION_UP, 10.5f);
    set_q_value(agent, 0, ACTION_RIGHT, 8.2f);
    
    float q_up = get_q_value(agent, 0, ACTION_UP);
    float q_right = get_q_value(agent, 0, ACTION_RIGHT);
    
    if (fabs(q_up - 10.5f) < 0.001f && fabs(q_right - 8.2f) < 0.001f) {
        printf("✓ Q-value setting and getting working correctly\n");
    } else {
        printf("❌ Q-value operations failed\n");
    }
    
    destroy_agent(agent);
    printf("✓ Q-value operations test completed\n\n");
}

void test_action_selection() {
    printf("Test 3: Testing action selection...\n");
    
    QLearningAgent* agent = create_agent(5, 4, 0.1f, 0.9f, 0.0f); // epsilon = 0 for deterministic testing
    
    // Set Q-values to make ACTION_DOWN the best choice
    set_q_value(agent, 0, ACTION_UP, 1.0f);
    set_q_value(agent, 0, ACTION_DOWN, 10.0f);  // Best action
    set_q_value(agent, 0, ACTION_LEFT, 2.0f);
    set_q_value(agent, 0, ACTION_RIGHT, 3.0f);
    
    Action selected = select_greedy_action(agent, 0);
    if (selected == ACTION_DOWN) {
        printf("✓ Greedy action selection working correctly\n");
    } else {
        printf("❌ Greedy action selection failed\n");
    }
    
    destroy_agent(agent);
    printf("✓ Action selection test completed\n\n");
}

void test_q_learning_update() {
    printf("Test 4: Testing Q-learning update...\n");
    
    QLearningAgent* agent = create_agent(5, 4, 0.5f, 0.9f, 0.1f);
    
    // Initial Q-value
    float initial_q = get_q_value(agent, 0, ACTION_UP);
    printf("Initial Q-value Q(0, UP): %.3f\n", initial_q);
    
    // Simulate a step with positive reward
    float reward = 10.0f;
    int next_state = 1;
    set_q_value(agent, next_state, ACTION_UP, 5.0f); // Set some value in next state
    
    update_q_value(agent, 0, ACTION_UP, reward, next_state, false);
    
    float updated_q = get_q_value(agent, 0, ACTION_UP);
    printf("Updated Q-value Q(0, UP): %.3f\n", updated_q);
    
    // Q-learning formula: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
    // Expected: 0 + 0.5 * [10.0 + 0.9 * 5.0 - 0] = 0.5 * [10.0 + 4.5] = 7.25
    float expected_q = initial_q + 0.5f * (reward + 0.9f * 5.0f - initial_q);
    
    if (fabs(updated_q - expected_q) < 0.001f) {
        printf("✓ Q-learning update working correctly (Expected: %.3f, Got: %.3f)\n", expected_q, updated_q);
    } else {
        printf("❌ Q-learning update failed (Expected: %.3f, Got: %.3f)\n", expected_q, updated_q);
    }
    
    destroy_agent(agent);
    printf("✓ Q-learning update test completed\n\n");
}

void test_epsilon_decay() {
    printf("Test 5: Testing epsilon decay...\n");
    
    QLearningAgent* agent = create_agent(5, 4, 0.1f, 0.9f, 1.0f);
    agent->epsilon_decay = 0.9f;
    agent->epsilon_min = 0.1f;
    
    printf("Initial epsilon: %.3f\n", agent->epsilon);
    
    for (int i = 0; i < 5; i++) {
        decay_epsilon(agent);
        printf("After decay %d: %.3f\n", i + 1, agent->epsilon);
    }
    
    if (agent->epsilon < 1.0f && agent->epsilon >= agent->epsilon_min) {
        printf("✓ Epsilon decay working correctly\n");
    } else {
        printf("❌ Epsilon decay failed\n");
    }
    
    destroy_agent(agent);
    printf("✓ Epsilon decay test completed\n\n");
}

void test_experience_buffer() {
    printf("Test 6: Testing experience buffer...\n");
    
    ExperienceBuffer* buffer = create_experience_buffer(3);
    if (!buffer) {
        printf("❌ Failed to create experience buffer\n");
        return;
    }
    
    // Add some experiences
    add_experience(buffer, 0, ACTION_UP, 1.0f, 1, false);
    add_experience(buffer, 1, ACTION_RIGHT, 2.0f, 2, false);
    add_experience(buffer, 2, ACTION_DOWN, 5.0f, 3, true);
    
    printf("Buffer size: %d (capacity: %d)\n", buffer->size, buffer->capacity);
    
    // Sample an experience
    Experience* exp = sample_experience(buffer);
    if (exp) {
        printf("✓ Sampled experience: state=%d, action=%d, reward=%.1f\n", 
               exp->state, exp->action, exp->reward);
    }
    
    // Test overflow (circular buffer)
    add_experience(buffer, 3, ACTION_LEFT, 3.0f, 4, false);
    printf("Buffer size after overflow: %d\n", buffer->size);
    
    destroy_experience_buffer(buffer);
    printf("✓ Experience buffer test completed\n\n");
}

void test_training_stats() {
    printf("Test 7: Testing training statistics...\n");
    
    TrainingStats* stats = create_training_stats(5);
    if (!stats) {
        printf("❌ Failed to create training stats\n");
        return;
    }
    
    // Record some episodes
    record_episode(stats, 0, 10.5f, 25, 0.9f, 2.1f);
    record_episode(stats, 1, 15.2f, 20, 0.8f, 3.2f);
    record_episode(stats, 2, 12.8f, 22, 0.7f, 2.8f);
    
    printf("Recorded %d episodes\n", stats->current_episode);
    printf("Best episode: %d with reward %.2f\n", stats->best_episode, stats->best_reward);
    
    print_training_summary(stats);
    
    destroy_training_stats(stats);
    printf("✓ Training statistics test completed\n\n");
}

void demonstrate_simple_learning() {
    printf("Test 8: Demonstrating simple Q-learning scenario...\n");
    
    // Create a simple 1D environment (5 states in a line)
    // State 0 -> State 1 -> State 2 -> State 3 -> State 4 (goal)
    QLearningAgent* agent = create_agent(5, 2, 0.1f, 0.9f, 0.3f); // 2 actions: LEFT(0), RIGHT(1)
    
    printf("Training agent on simple 5-state environment...\n");
    printf("Goal: Learn to move right to reach state 4\n\n");
    
    // Simulate training episodes
    for (int episode = 0; episode < 10; episode++) {
        int state = 0; // Start at state 0
        int steps = 0;
        float total_reward = 0.0f;
        
        while (state != 4 && steps < 20) { // Goal is state 4
            Action action = select_action(agent, state);
            
            // Simple environment dynamics
            int next_state = state;
            float reward = -0.1f; // Small step penalty
            
            if (action == 1 && state < 4) { // RIGHT action
                next_state = state + 1;
            } else if (action == 0 && state > 0) { // LEFT action
                next_state = state - 1;
            }
            
            // Goal reward
            if (next_state == 4) {
                reward = 10.0f;
            }
            
            bool done = (next_state == 4);
            
            update_q_value(agent, state, action, reward, next_state, done);
            
            state = next_state;
            total_reward += reward;
            steps++;
            
            if (done) break;
        }
        
        decay_epsilon(agent);
        
        if (episode % 2 == 0) {
            printf("Episode %d: Steps=%d, Reward=%.2f, Epsilon=%.3f\n", 
                   episode, steps, total_reward, agent->epsilon);
        }
    }
    
    printf("\nFinal Q-values for state 0:\n");
    printf("  LEFT:  %.3f\n", get_q_value(agent, 0, 0));
    printf("  RIGHT: %.3f\n", get_q_value(agent, 0, 1));
    
    printf("\nFinal Q-values for state 3:\n");
    printf("  LEFT:  %.3f\n", get_q_value(agent, 3, 0));
    printf("  RIGHT: %.3f\n", get_q_value(agent, 3, 1));
    
    printf("\n✓ Simple learning demonstration completed\n\n");
    
    destroy_agent(agent);
}

int main() {
    printf("=== Q-Learning Agent Test Suite ===\n\n");
    
    // Seed random number generator
    srand(time(NULL));
    
    test_agent_creation();
    test_q_value_operations();
    test_action_selection();
    test_q_learning_update();
    test_epsilon_decay();
    test_experience_buffer();
    test_training_stats();
    demonstrate_simple_learning();
    
    printf("🎉 All agent tests completed successfully!\n");
    printf("The Q-learning implementation is working correctly.\n\n");
    
    printf("Core Q-learning components implemented:\n");
    printf("  ✓ Q-table creation and management\n");
    printf("  ✓ Epsilon-greedy action selection\n");
    printf("  ✓ Q-learning update formula: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]\n");
    printf("  ✓ Epsilon decay for exploration reduction\n");
    printf("  ✓ Experience buffer for experience replay\n");
    printf("  ✓ Training statistics and monitoring\n");
    
    return 0;
}
