#include "rendering.h"
#include "environment.h"
#include "agent.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

// Training configuration
typedef struct {
    int num_episodes;
    int max_steps_per_episode;
    bool enable_visualization;
    bool save_policy;
    bool print_progress;
    int progress_interval;
    const char* policy_filename;
} TrainingConfig;

// Function to save learned policy to file
void save_policy_to_file(QLearningAgent* agent, GridWorld* world, const char* filename) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        printf("Error: Could not open file %s for writing\n", filename);
        return;
    }
    
    fprintf(file, "# Q-Learning Policy\n");
    fprintf(file, "# Grid dimensions: %dx%d\n", world->width, world->height);
    fprintf(file, "# States: %d, Actions: %d\n", agent->num_states, agent->num_actions);
    fprintf(file, "# Format: state_x,state_y,action_up,action_down,action_left,action_right,best_action\n");
    
    for (int state = 0; state < agent->num_states; state++) {
        Position pos = state_to_position(world, state);
        
        // Skip wall states
        if (!is_walkable(world, pos.x, pos.y)) {
            continue;
        }
        
        fprintf(file, "%d,%d", pos.x, pos.y);
        
        // Write Q-values for all actions
        for (int action = 0; action < agent->num_actions; action++) {
            fprintf(file, ",%.3f", get_q_value(agent, state, action));
        }
        
        // Find and write best action
        Action best_action = select_greedy_action(agent, state);
        fprintf(file, ",%d\n", best_action);
    }
    
    fclose(file);
    printf("Policy saved to %s\n", filename);
}

// Function to print training progress
void print_episode_progress(int episode, EpisodeStats* stats, QLearningAgent* agent) {
    printf("Episode %d: Reward=%.2f, Steps=%d, Epsilon=%.3f, Avg Q=%.3f\n",
           stats->episode, stats->total_reward, stats->steps_taken, 
           stats->epsilon_used, stats->avg_q_value);
}

// Function to calculate average Q-value for current state
float calculate_avg_q_value(QLearningAgent* agent, int state) {
    float sum = 0.0f;
    for (int action = 0; action < agent->num_actions; action++) {
        sum += get_q_value(agent, state, action);
    }
    return sum / agent->num_actions;
}

// Main training function
void run_training(GridWorld* world, QLearningAgent* agent, TrainingConfig* config) {
    printf("Starting Q-Learning Training...\n");
    printf("Episodes: %d, Max steps per episode: %d\n", 
           config->num_episodes, config->max_steps_per_episode);
    printf("Visualization: %s\n", config->enable_visualization ? "ON" : "OFF");
    
    // Initialize training statistics
    TrainingStats* stats = create_training_stats(config->num_episodes);
    if (!stats) {
        printf("Error: Failed to create training statistics\n");
        return;
    }
    
    // Initialize visualization if enabled
    VisualizationState* vis_state = NULL;
    if (config->enable_visualization) {
        const int SCREEN_WIDTH = 800;
        const int SCREEN_HEIGHT = 600;
        const int CELL_SIZE = 40;
        
        init_graphics(SCREEN_WIDTH, SCREEN_HEIGHT);
        vis_state = get_visualization_state();
        
        printf("Visualization enabled. Controls:\n");
        printf("  Q - Toggle Q-value visualization\n");
        printf("  G - Toggle grid lines\n");
        printf("  P - Pause/Resume training\n");
        printf("  ESC - Exit training\n");
    }
    
    bool training_paused = false;
    clock_t start_time = clock();
    
    // Main training loop
    for (int episode = 0; episode < config->num_episodes; episode++) {
        // Reset environment for new episode
        reset_environment(world);
        
        float episode_reward = 0.0f;
        int steps_taken = 0;
        float total_q_value = 0.0f;
        int q_value_count = 0;
        
        // Episode loop
        while (!world->episode_done && steps_taken < config->max_steps_per_episode) {
            // Handle visualization input
            if (config->enable_visualization) {
                if (WindowShouldClose()) {
                    printf("Training interrupted by user\n");
                    goto training_cleanup;
                }
                
                if (IsKeyPressed(KEY_P)) {
                    training_paused = !training_paused;
                    printf("Training %s\n", training_paused ? "PAUSED" : "RESUMED");
                }
                
                if (IsKeyPressed(KEY_Q) && vis_state) {
                    vis_state->config.show_q_values = !vis_state->config.show_q_values;
                }
                
                if (IsKeyPressed(KEY_G) && vis_state) {
                    vis_state->config.show_grid = !vis_state->config.show_grid;
                }
                
                if (training_paused) {
                    // Still render while paused
                    BeginDrawing();
                    ClearBackground(RAYWHITE);
                    
                    if (vis_state->config.show_q_values) {
                        draw_q_values(vis_state, world, agent);
                    } else {
                        draw_grid_world(vis_state, world);
                    }
                    
                    draw_walls(vis_state, world);
                    draw_goal(vis_state, world->goal_pos);
                    draw_agent(vis_state, world->agent_pos);
                    
                    DrawText("TRAINING PAUSED - Press P to resume", 10, 10, 20, RED);
                    
                    EndDrawing();
                    continue;
                }
            }
            
            // Get current state
            int current_state = get_state_index(world);
            
            // Select action using epsilon-greedy policy
            Action action = select_action(agent, current_state);
            
            // Take action and get result
            StepResult result = step_environment(world, action);
            
            // Update Q-value
            update_q_value(agent, current_state, action, result.reward, 
                          position_to_state(world, result.next_state.position), result.done);
            
            // Accumulate statistics
            episode_reward += result.reward;
            steps_taken++;
            
            // Calculate average Q-value for statistics
            float avg_q = calculate_avg_q_value(agent, current_state);
            total_q_value += avg_q;
            q_value_count++;
            
            // Render if visualization is enabled
            if (config->enable_visualization && !training_paused) {
                BeginDrawing();
                ClearBackground(RAYWHITE);
                
                // Draw environment
                if (vis_state->config.show_q_values) {
                    draw_q_values(vis_state, world, agent);
                } else {
                    draw_grid_world(vis_state, world);
                }
                
                draw_walls(vis_state, world);
                draw_goal(vis_state, world->goal_pos);
                draw_agent(vis_state, world->agent_pos);
                
                // Draw training info
                char info_text[512];
                snprintf(info_text, sizeof(info_text), 
                        "Episode: %d/%d | Step: %d | Reward: %.1f | Epsilon: %.3f",
                        episode + 1, config->num_episodes, steps_taken, episode_reward, agent->epsilon);
                DrawText(info_text, 10, 10, 16, BLACK);
                
                snprintf(info_text, sizeof(info_text),
                        "Agent: (%d,%d) | Action: %s | Q-values: %s",
                        world->agent_pos.x, world->agent_pos.y,
                        action == ACTION_UP ? "UP" : action == ACTION_DOWN ? "DOWN" : 
                        action == ACTION_LEFT ? "LEFT" : "RIGHT",
                        vis_state->config.show_q_values ? "ON" : "OFF");
                DrawText(info_text, 10, 30, 14, DARKGRAY);
                
                DrawText("P: Pause | Q: Toggle Q-values | G: Toggle Grid | ESC: Exit", 
                        10, GetScreenHeight() - 30, 12, DARKBLUE);
                
                EndDrawing();
                
                // Control training speed for visualization
                WaitTime(0.05f); // 50ms delay for visibility
            }
        }
        
        // Decay epsilon after each episode
        decay_epsilon(agent);
        
        // Record episode statistics
        float avg_q_episode = q_value_count > 0 ? total_q_value / q_value_count : 0.0f;
        record_episode(stats, episode, episode_reward, steps_taken, agent->epsilon, avg_q_episode);
        
        // Print progress
        if (config->print_progress && (episode + 1) % config->progress_interval == 0) {
            EpisodeStats* episode_stats = &stats->episodes[episode];
            print_episode_progress(episode + 1, episode_stats, agent);
        }
    }
    
    training_cleanup:
    {
        clock_t end_time = clock();
        double training_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
        
        printf("\nTraining completed!\n");
        printf("Total training time: %.2f seconds\n", training_time);
        print_training_summary(stats);
        
        // Save policy if requested
        if (config->save_policy && config->policy_filename) {
            save_policy_to_file(agent, world, config->policy_filename);
        }
        
        // Cleanup
        destroy_training_stats(stats);
        if (config->enable_visualization) {
            cleanup_graphics();
        }
    }
}

// Function to create default training configuration
TrainingConfig create_default_training_config() {
    TrainingConfig config = {
        .num_episodes = 1000,
        .max_steps_per_episode = 200,
        .enable_visualization = false,
        .save_policy = true,
        .print_progress = true,
        .progress_interval = 100,
        .policy_filename = "learned_policy.txt"
    };
    return config;
}

// Function to parse command line arguments
TrainingConfig parse_arguments(int argc, char* argv[]) {
    TrainingConfig config = create_default_training_config();
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--episodes") == 0 && i + 1 < argc) {
            config.num_episodes = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--max-steps") == 0 && i + 1 < argc) {
            config.max_steps_per_episode = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--visualize") == 0) {
            config.enable_visualization = true;
        } else if (strcmp(argv[i], "--no-save") == 0) {
            config.save_policy = false;
        } else if (strcmp(argv[i], "--quiet") == 0) {
            config.print_progress = false;
        } else if (strcmp(argv[i], "--policy-file") == 0 && i + 1 < argc) {
            config.policy_filename = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0) {
            printf("Q-Learning Training Options:\n");
            printf("  --episodes N        Number of training episodes (default: 1000)\n");
            printf("  --max-steps N       Maximum steps per episode (default: 200)\n");
            printf("  --visualize         Enable real-time visualization\n");
            printf("  --no-save           Don't save learned policy\n");
            printf("  --quiet             Don't print progress during training\n");
            printf("  --policy-file FILE  Filename for saved policy (default: learned_policy.txt)\n");
            printf("  --help              Show this help message\n");
            exit(0);
        }
    }
    
    return config;
}

int main(int argc, char* argv[]) {
    // Initialize random seed
    srand((unsigned int)time(NULL));
    
    // Parse command line arguments
    TrainingConfig config = parse_arguments(argc, argv);
    
    printf("Q-Learning Agent Training\n");
    printf("========================\n");
    
    // Environment configuration
    const int GRID_WIDTH = 10;
    const int GRID_HEIGHT = 10;
    
    // Create environment
    GridWorld* world = create_grid_world(GRID_WIDTH, GRID_HEIGHT);
    if (!world) {
        printf("Error: Failed to create grid world\n");
        return -1;
    }
    
    // Configure environment
    world->start_pos = (Position){1, 1};
    world->goal_pos = (Position){8, 8};
    world->step_penalty = -0.1f;
    world->goal_reward = 100.0f;
    world->wall_penalty = -10.0f;
    world->max_steps = config.max_steps_per_episode;
    
    // Set up environment layout
    printf("Setting up environment...\n");
    
    // Add some walls to make it interesting
    set_cell(world, 3, 3, CELL_WALL);
    set_cell(world, 3, 4, CELL_WALL);
    set_cell(world, 3, 5, CELL_WALL);
    set_cell(world, 5, 2, CELL_WALL);
    set_cell(world, 5, 3, CELL_WALL);
    set_cell(world, 5, 4, CELL_WALL);
    set_cell(world, 7, 6, CELL_WALL);
    set_cell(world, 7, 7, CELL_WALL);
    
    // Mark goal and start positions in the grid
    set_cell(world, world->goal_pos.x, world->goal_pos.y, CELL_GOAL);
    set_cell(world, world->start_pos.x, world->start_pos.y, CELL_START);
    
    // Create Q-learning agent
    int num_states = GRID_WIDTH * GRID_HEIGHT;
    QLearningAgent* agent = create_agent(num_states, NUM_ACTIONS, 0.1f, 0.9f, 1.0f);
    if (!agent) {
        printf("Error: Failed to create agent\n");
        destroy_grid_world(world);
        return -1;
    }
    
    // Set epsilon decay
    agent->epsilon_decay = 0.995f;
    agent->epsilon_min = 0.01f;
    
    printf("Agent created with parameters:\n");
    printf("  Learning rate: %.3f\n", agent->learning_rate);
    printf("  Discount factor: %.3f\n", agent->discount_factor);
    printf("  Initial epsilon: %.3f\n", agent->epsilon);
    printf("  Epsilon decay: %.3f\n", agent->epsilon_decay);
    printf("  Minimum epsilon: %.3f\n", agent->epsilon_min);
    
    // Validate environment
    if (!validate_environment(world)) {
        printf("Error: Invalid environment configuration\n");
        destroy_agent(agent);
        destroy_grid_world(world);
        return -1;
    }
    
    print_environment_info(world);
    
    // Run training
    run_training(world, agent, &config);
    
    printf("\nTraining session completed successfully!\n");
    
    // Cleanup
    destroy_agent(agent);
    destroy_grid_world(world);
    
    return 0;
}
