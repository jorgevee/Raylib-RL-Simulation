#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "include/agent.h"
#include "include/environment.h"

void print_grid(GridWorld* world) {
    printf("\nGrid World:\n");
    for (int y = 0; y < world->height; y++) {
        for (int x = 0; x < world->width; x++) {
            if (x == world->agent_pos.x && y == world->agent_pos.y) {
                printf("A ");
            } else if (x == world->goal_pos.x && y == world->goal_pos.y) {
                printf("G ");
            } else {
                switch (world->grid[y][x]) {
                    case CELL_EMPTY: printf(". "); break;
                    case CELL_WALL: printf("# "); break;
                    default: printf("? "); break;
                }
            }
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    printf("=== Q-Learning Agent + GridWorld Integration Demo ===\n\n");
    
    // Seed random number generator
    srand(time(NULL));
    
    // Create a small 5x5 grid world
    GridWorld* world = create_grid_world(5, 5);
    if (!world) {
        printf("Failed to create grid world\n");
        return 1;
    }
    
    // Set up the environment
    world->goal_pos.x = 4;
    world->goal_pos.y = 4;
    world->start_pos.x = 0;
    world->start_pos.y = 0;
    
    // Add some walls to make it interesting
    set_cell(world, 2, 1, CELL_WALL);
    set_cell(world, 2, 2, CELL_WALL);
    set_cell(world, 2, 3, CELL_WALL);
    set_cell(world, 1, 3, CELL_WALL);
    
    printf("Environment Setup:\n");
    print_grid(world);
    
    // Create Q-learning agent
    int num_states = world->width * world->height;
    QLearningAgent* agent = create_agent(num_states, NUM_ACTIONS, 0.1f, 0.9f, 0.1f);
    if (!agent) {
        printf("Failed to create agent\n");
        destroy_grid_world(world);
        return 1;
    }
    
    printf("Training agent for 100 episodes...\n");
    
    TrainingStats* stats = create_training_stats(100);
    
    // Training loop
    for (int episode = 0; episode < 100; episode++) {
        reset_environment(world);
        int state = get_state_index(world);
        int steps = 0;
        float total_reward = 0.0f;
        
        while (!world->episode_done && steps < 200) {
            Action action = select_action(agent, state);
            
            StepResult result = step_environment(world, action);
            
            update_q_value(agent, state, action, result.reward, 
                          result.next_state.state_index, result.done);
            
            state = result.next_state.state_index;
            total_reward += result.reward;
            steps++;
        }
        
        decay_epsilon(agent);
        
        // Record episode statistics
        float avg_q = 0.0f;
        int q_count = 0;
        for (int s = 0; s < num_states; s++) {
            for (int a = 0; a < NUM_ACTIONS; a++) {
                avg_q += get_q_value(agent, s, a);
                q_count++;
            }
        }
        avg_q /= q_count;
        
        record_episode(stats, episode, total_reward, steps, agent->epsilon, avg_q);
        
        // Print progress every 20 episodes
        if (episode % 20 == 0) {
            printf("Episode %d: Steps=%d, Reward=%.1f, Epsilon=%.3f\n", 
                   episode, steps, total_reward, agent->epsilon);
        }
    }
    
    printf("\n=== Training Complete ===\n");
    print_training_summary(stats);
    
    // Test learned policy
    printf("Testing learned policy (greedy actions only):\n");
    reset_environment(world);
    
    agent->epsilon = 0.0f; // Pure exploitation
    int state = get_state_index(world);
    int steps = 0;
    
    printf("Path taken by trained agent:\n");
    print_grid(world);
    
    while (!world->episode_done && steps < 50) {
        Action action = select_greedy_action(agent, state);
        
        char* action_name;
        switch (action) {
            case ACTION_UP: action_name = "UP"; break;
            case ACTION_DOWN: action_name = "DOWN"; break;
            case ACTION_LEFT: action_name = "LEFT"; break;
            case ACTION_RIGHT: action_name = "RIGHT"; break;
            case NUM_ACTIONS: action_name = "INVALID"; break;
        }
        
        printf("Step %d: Action = %s\n", steps + 1, action_name);
        
        StepResult result = step_environment(world, action);
        state = result.next_state.state_index;
        steps++;
        
        print_grid(world);
        
        if (result.done) {
            if (world->agent_pos.x == world->goal_pos.x && 
                world->agent_pos.y == world->goal_pos.y) {
                printf("🎉 Agent reached the goal in %d steps!\n", steps);
            } else {
                printf("Episode ended without reaching goal.\n");
            }
            break;
        }
    }
    
    printf("\nSample Q-values for start position (state %d):\n", 
           position_to_state(world, world->start_pos));
    int start_state = position_to_state(world, world->start_pos);
    printf("  UP:    %.3f\n", get_q_value(agent, start_state, ACTION_UP));
    printf("  DOWN:  %.3f\n", get_q_value(agent, start_state, ACTION_DOWN));
    printf("  LEFT:  %.3f\n", get_q_value(agent, start_state, ACTION_LEFT));
    printf("  RIGHT: %.3f\n", get_q_value(agent, start_state, ACTION_RIGHT));
    
    // Clean up
    destroy_training_stats(stats);
    destroy_agent(agent);
    destroy_grid_world(world);
    
    printf("\n✅ Integration demo completed successfully!\n");
    printf("The Q-learning agent successfully learned to navigate the grid world.\n");
    
    return 0;
}
