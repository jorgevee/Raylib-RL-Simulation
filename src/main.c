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

// Training control state
typedef struct {
    bool is_paused;
    bool should_reset;
    bool should_exit;
    bool show_q_values;
    float training_speed;
    bool save_requested;
    bool load_requested;
    const char* qtable_filename;
} TrainingControl;

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

// Initialize training control state
TrainingControl create_training_control() {
    TrainingControl control = {
        .is_paused = false,
        .should_reset = false,
        .should_exit = false,
        .show_q_values = false,
        .training_speed = 1.0f,
        .save_requested = false,
        .load_requested = false,
        .qtable_filename = "qtable.dat"
    };
    return control;
}

// Handle user input for training controls
void handle_training_input(TrainingControl* control, VisualizationState* vis_state) {
    // Space: Pause/Resume training
    if (IsKeyPressed(KEY_SPACE)) {
        control->is_paused = !control->is_paused;
        printf("Training %s\n", control->is_paused ? "PAUSED" : "RESUMED");
    }
    
    // R: Reset and restart
    if (IsKeyPressed(KEY_R)) {
        control->should_reset = true;
        printf("Training RESET requested\n");
    }
    
    // V: Toggle Q-value visualization
    if (IsKeyPressed(KEY_V)) {
        control->show_q_values = !control->show_q_values;
        if (vis_state) {
            vis_state->config.show_q_values = control->show_q_values;
        }
        printf("Q-value visualization: %s\n", control->show_q_values ? "ON" : "OFF");
    }
    
    // +: Increase training speed
    if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD)) {
        control->training_speed = fminf(control->training_speed * 1.5f, 10.0f);
        printf("Training speed: %.1fx\n", control->training_speed);
    }
    
    // -: Decrease training speed
    if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT)) {
        control->training_speed = fmaxf(control->training_speed / 1.5f, 0.1f);
        printf("Training speed: %.1fx\n", control->training_speed);
    }
    
    // S: Save current Q-table
    if (IsKeyPressed(KEY_S)) {
        control->save_requested = true;
        printf("Q-table save requested\n");
    }
    
    // L: Load saved Q-table
    if (IsKeyPressed(KEY_L)) {
        control->load_requested = true;
        printf("Q-table load requested\n");
    }
    
    // ESC: Exit training
    if (IsKeyPressed(KEY_ESCAPE) || WindowShouldClose()) {
        control->should_exit = true;
        printf("Training exit requested\n");
    }
    
    // Additional visualization controls
    if (IsKeyPressed(KEY_Q) && vis_state) {
        vis_state->config.show_q_values = !vis_state->config.show_q_values;
        control->show_q_values = vis_state->config.show_q_values;
    }
    
    if (IsKeyPressed(KEY_G) && vis_state) {
        vis_state->config.show_grid = !vis_state->config.show_grid;
    }
}

// Display control instructions
void display_control_instructions() {
    printf("\n=== Training Controls ===\n");
    printf("SPACE   : Pause/Resume training\n");
    printf("R       : Reset and restart training\n");
    printf("V       : Toggle Q-value visualization\n");
    printf("+/=     : Increase training speed\n");
    printf("-       : Decrease training speed\n");
    printf("S       : Save current Q-table\n");
    printf("L       : Load saved Q-table\n");
    printf("Q       : Toggle Q-value display\n");
    printf("G       : Toggle grid lines\n");
    printf("ESC     : Exit training\n");
    printf("========================\n\n");
}

// Main training function with enhanced controls
void run_training(GridWorld* world, QLearningAgent* agent, TrainingConfig* config) {
    printf("Starting Q-Learning Training with Enhanced Controls...\n");
    printf("Episodes: %d, Max steps per episode: %d\n", 
           config->num_episodes, config->max_steps_per_episode);
    printf("Visualization: %s\n", config->enable_visualization ? "ON" : "OFF");
    
    // Display control instructions
    if (config->enable_visualization) {
        display_control_instructions();
    }
    
    // Initialize training control
    TrainingControl control = create_training_control();
    
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
    }
    
    clock_t start_time = clock();
    int episode = 0;
    
    // Main training loop
    while (episode < config->num_episodes && !control.should_exit) {
        // Handle reset request
        if (control.should_reset) {
            printf("Resetting training...\n");
            episode = 0;
            
            // Reset agent (clear Q-table)
            for (int s = 0; s < agent->num_states; s++) {
                for (int a = 0; a < agent->num_actions; a++) {
                    agent->q_table[s][a] = 0.0f;
                }
            }
            
            // Reset epsilon
            agent->epsilon = 1.0f;
            
            // Clear statistics
            destroy_training_stats(stats);
            stats = create_training_stats(config->num_episodes);
            
            control.should_reset = false;
            start_time = clock();
            printf("Training reset complete!\n");
        }
        
        // Reset environment for new episode
        reset_environment(world);
        
        float episode_reward = 0.0f;
        int steps_taken = 0;
        float total_q_value = 0.0f;
        int q_value_count = 0;
        
        // Episode loop
        while (!world->episode_done && steps_taken < config->max_steps_per_episode && !control.should_exit) {
            // Handle user input
            if (config->enable_visualization) {
                handle_training_input(&control, vis_state);
                
                // Handle save request
                if (control.save_requested) {
                    if (save_q_table(agent, control.qtable_filename)) {
                        printf("Q-table saved successfully!\n");
                    }
                    control.save_requested = false;
                }
                
                // Handle load request
                if (control.load_requested) {
                    if (load_q_table(agent, control.qtable_filename)) {
                        printf("Q-table loaded successfully!\n");
                    }
                    control.load_requested = false;
                }
                
                // Handle exit request
                if (control.should_exit) {
                    printf("Training interrupted by user\n");
                    goto training_cleanup;
                }
                
                // Skip training step if paused, but continue rendering
                if (control.is_paused) {
                    BeginDrawing();
                    ClearBackground(RAYWHITE);
                    
                    if (control.show_q_values) {
                        draw_q_values(vis_state, world, agent);
                    } else {
                        draw_grid_world(vis_state, world);
                    }
                    
                    draw_walls(vis_state, world);
                    draw_goal(vis_state, world->goal_pos);
                    draw_agent(vis_state, world->agent_pos);
                    
                    // Draw enhanced status
                    char status_text[256];
                    snprintf(status_text, sizeof(status_text), 
                            "TRAINING PAUSED - Episode: %d/%d | Speed: %.1fx", 
                            episode + 1, config->num_episodes, control.training_speed);
                    DrawText(status_text, 10, 10, 20, RED);
                    
                    DrawText("SPACE: Resume | R: Reset | V: Q-values | S: Save | L: Load | ESC: Exit", 
                            10, GetScreenHeight() - 50, 12, DARKBLUE);
                    
                    EndDrawing();
                    WaitTime(0.016f); // ~60 FPS for smooth UI
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
            if (config->enable_visualization) {
                BeginDrawing();
                ClearBackground(RAYWHITE);
                
                // Draw environment
                if (control.show_q_values) {
                    draw_q_values(vis_state, world, agent);
                } else {
                    draw_grid_world(vis_state, world);
                }
                
                draw_walls(vis_state, world);
                draw_goal(vis_state, world->goal_pos);
                draw_agent(vis_state, world->agent_pos);
                
                // Draw enhanced training info
                char info_text[512];
                snprintf(info_text, sizeof(info_text), 
                        "Episode: %d/%d | Step: %d | Reward: %.1f | Epsilon: %.3f | Speed: %.1fx",
                        episode + 1, config->num_episodes, steps_taken, episode_reward, 
                        agent->epsilon, control.training_speed);
                DrawText(info_text, 10, 10, 16, BLACK);
                
                snprintf(info_text, sizeof(info_text),
                        "Agent: (%d,%d) | Action: %s | Q-values: %s",
                        world->agent_pos.x, world->agent_pos.y,
                        action == ACTION_UP ? "UP" : action == ACTION_DOWN ? "DOWN" : 
                        action == ACTION_LEFT ? "LEFT" : "RIGHT",
                        control.show_q_values ? "ON" : "OFF");
                DrawText(info_text, 10, 30, 14, DARKGRAY);
                
                DrawText("SPACE: Pause | R: Reset | V: Q-values | +/-: Speed | S: Save | L: Load | ESC: Exit", 
                        10, GetScreenHeight() - 30, 12, DARKBLUE);
                
                EndDrawing();
                
                // Control training speed
                float delay = 0.05f / control.training_speed;
                WaitTime(delay);
            }
        }
        
        // Decay epsilon after each episode
        decay_epsilon(agent);
        
        // Calculate Q-value variance for performance metrics
        float q_variance = calculate_q_value_variance(agent);
        
        // Check if goal was reached
        bool goal_reached = (world->agent_pos.x == world->goal_pos.x && 
                            world->agent_pos.y == world->goal_pos.y);
        
        // Record episode statistics
        float avg_q_episode = q_value_count > 0 ? total_q_value / q_value_count : 0.0f;
        record_episode(stats, episode, episode_reward, steps_taken, agent->epsilon, avg_q_episode);
        
        // Update performance metrics
        update_performance_metrics(stats->metrics, stats, episode, goal_reached, q_variance);
        
        // Check for convergence
        bool converged = check_convergence(stats->metrics, episode);
        if (converged && !config->enable_visualization) {
            printf("Training converged at episode %d!\n", episode + 1);
        }
        
        // Print progress with enhanced metrics
        if (config->print_progress && (episode + 1) % config->progress_interval == 0) {
            EpisodeStats* episode_stats = &stats->episodes[episode];
            print_episode_progress(episode + 1, episode_stats, agent);
            
            // Print learning curves every 200 episodes
            if ((episode + 1) % (config->progress_interval * 2) == 0) {
                print_learning_curves(stats, 20);
            }
            
            // Print convergence analysis every 100 episodes
            if ((episode + 1) % config->progress_interval == 0) {
                print_convergence_analysis(stats->metrics, episode);
            }
        }
        
        episode++;
    }
    
    training_cleanup:
    {
        clock_t end_time = clock();
        double training_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
        
        printf("\nTraining completed!\n");
        printf("Total training time: %.2f seconds\n", training_time);
        printf("Final training speed: %.1fx\n", control.training_speed);
        
        // Print final performance analysis
        print_training_summary(stats);
        print_learning_curves(stats, 50);  // Show last 50 episodes
        print_convergence_analysis(stats->metrics, episode - 1);
        
        // Save performance data
        save_performance_data(stats, "performance_data.csv");
        
        // Save policy if requested
        if (config->save_policy && config->policy_filename) {
            save_policy_to_file(agent, world, config->policy_filename);
        }
        
        // Auto-save Q-table at end
        if (config->enable_visualization) {
            if (save_q_table(agent, control.qtable_filename)) {
                printf("Q-table auto-saved to %s\n", control.qtable_filename);
            }
        }
        
        // Final performance summary
        if (stats->metrics && stats->current_episode > 0) {
            printf("\n=== Final Performance Summary ===\n");
            printf("Episodes completed: %d\n", stats->current_episode);
            printf("Best episode: %d (reward: %.2f)\n", stats->best_episode + 1, stats->best_reward);
            printf("Training converged: %s\n", stats->metrics->has_converged ? "Yes" : "No");
            if (stats->metrics->has_converged) {
                printf("Convergence episode: %d\n", stats->metrics->convergence_episode + 1);
            }
            
            // Calculate final success rate
            int successful_episodes = 0;
            for (int i = 0; i < stats->current_episode; i++) {
                if (stats->metrics->success_episodes[i]) {
                    successful_episodes++;
                }
            }
            float final_success_rate = (float)successful_episodes / stats->current_episode * 100.0f;
            printf("Overall success rate: %.1f%% (%d/%d episodes)\n", 
                   final_success_rate, successful_episodes, stats->current_episode);
            printf("==================================\n");
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
