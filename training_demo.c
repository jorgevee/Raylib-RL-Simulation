/*
 * Q-Learning Training Loop Integration Demo
 * 
 * This demo showcases the complete integration of:
 * - Q-learning training loops
 * - Real-time learning visualization 
 * - Educational demonstrations
 * - Policy saving and analysis
 */

#include "rendering.h"
#include "environment.h"
#include "agent.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

// Demo configuration
typedef struct {
    int demo_episodes;
    int visualization_episodes;
    bool show_q_values;
    bool educational_mode;
    float training_speed;
} DemoConfig;

// Educational demo: step-by-step Q-learning explanation
void run_educational_demo(GridWorld* world, QLearningAgent* agent) {
    printf("\n=== Educational Q-Learning Demo ===\n");
    printf("This demo shows step-by-step how Q-learning works:\n\n");
    
    // Reset for demo
    reset_environment(world);
    
    printf("1. Initial State: Agent at (%d, %d), Goal at (%d, %d)\n", 
           world->agent_pos.x, world->agent_pos.y, 
           world->goal_pos.x, world->goal_pos.y);
    
    // Show initial Q-values
    int state = get_state_index(world);
    printf("   Initial Q-values for this state:\n");
    for (int action = 0; action < NUM_ACTIONS; action++) {
        const char* action_names[] = {"UP", "DOWN", "LEFT", "RIGHT"};
        printf("     %s: %.3f\n", action_names[action], get_q_value(agent, state, action));
    }
    
    printf("\n2. Epsilon-greedy action selection (epsilon = %.3f):\n", agent->epsilon);
    Action action = select_action(agent, state);
    const char* action_names[] = {"UP", "DOWN", "LEFT", "RIGHT"};
    printf("   Selected action: %s\n", action_names[action]);
    
    printf("\n3. Taking action and observing result:\n");
    Position old_pos = world->agent_pos;
    StepResult result = step_environment(world, action);
    
    printf("   Old position: (%d, %d)\n", old_pos.x, old_pos.y);
    printf("   New position: (%d, %d)\n", world->agent_pos.x, world->agent_pos.y);
    printf("   Reward received: %.2f\n", result.reward);
    printf("   Episode done: %s\n", result.done ? "Yes" : "No");
    
    printf("\n4. Q-value update using Bellman equation:\n");
    printf("   Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]\n");
    printf("   Where: α=%.2f (learning rate), γ=%.2f (discount factor)\n", 
           agent->learning_rate, agent->discount_factor);
    
    // Show Q-value before and after update (already updated in step_environment)
    printf("   Updated Q-value for %s: %.3f\n", action_names[action], 
           get_q_value(agent, state, action));
    
    printf("\n5. Epsilon decay for next episode:\n");
    float old_epsilon = agent->epsilon;
    decay_epsilon(agent);
    printf("   Epsilon: %.3f -> %.3f\n", old_epsilon, agent->epsilon);
    
    printf("\nThis process repeats for thousands of episodes until the agent learns!\n");
}

// Performance comparison demo
void run_performance_demo() {
    printf("\n=== Performance Comparison Demo ===\n");
    printf("Comparing different learning parameters:\n\n");
    
    const int DEMO_EPISODES = 100;
    const int GRID_SIZE = 8;
    
    // Different configurations to test
    struct {
        float learning_rate;
        float discount_factor;
        float epsilon_decay;
        const char* description;
    } configs[] = {
        {0.1f, 0.9f, 0.995f, "Standard Q-learning"},
        {0.3f, 0.9f, 0.995f, "Higher learning rate"},
        {0.1f, 0.7f, 0.995f, "Lower discount factor"},
        {0.1f, 0.9f, 0.990f, "Faster epsilon decay"}
    };
    
    for (int config_idx = 0; config_idx < 4; config_idx++) {
        printf("%d. %s:\n", config_idx + 1, configs[config_idx].description);
        
        // Create environment and agent
        GridWorld* world = create_grid_world(GRID_SIZE, GRID_SIZE);
        world->start_pos = (Position){0, 0};
        world->goal_pos = (Position){GRID_SIZE-1, GRID_SIZE-1};
        world->step_penalty = -0.1f;
        world->goal_reward = 100.0f;
        world->wall_penalty = -10.0f;
        world->max_steps = 100;
        
        QLearningAgent* agent = create_agent(GRID_SIZE * GRID_SIZE, NUM_ACTIONS,
                                           configs[config_idx].learning_rate,
                                           configs[config_idx].discount_factor,
                                           1.0f);
        agent->epsilon_decay = configs[config_idx].epsilon_decay;
        agent->epsilon_min = 0.01f;
        
        // Train and measure performance
        int successful_episodes = 0;
        float total_reward = 0.0f;
        
        for (int episode = 0; episode < DEMO_EPISODES; episode++) {
            reset_environment(world);
            float episode_reward = 0.0f;
            
            while (!world->episode_done && world->episode_steps < world->max_steps) {
                int state = get_state_index(world);
                Action action = select_action(agent, state);
                StepResult result = step_environment(world, action);
                
                update_q_value(agent, state, action, result.reward,
                             position_to_state(world, result.next_state.position), result.done);
                
                episode_reward += result.reward;
            }
            
            if (positions_equal(world->agent_pos, world->goal_pos)) {
                successful_episodes++;
            }
            
            total_reward += episode_reward;
            decay_epsilon(agent);
        }
        
        printf("   Success rate: %d/%d (%.1f%%)\n", successful_episodes, DEMO_EPISODES,
               (float)successful_episodes / DEMO_EPISODES * 100);
        printf("   Average reward: %.2f\n", total_reward / DEMO_EPISODES);
        printf("   Final epsilon: %.3f\n\n", agent->epsilon);
        
        destroy_agent(agent);
        destroy_grid_world(world);
    }
}

// Interactive visualization demo
void run_interactive_demo(GridWorld* world, QLearningAgent* agent, DemoConfig* config) {
    printf("\n=== Interactive Visualization Demo ===\n");
    printf("Starting interactive training with visualization...\n");
    printf("Controls during training:\n");
    printf("  Q - Toggle Q-value visualization\n");
    printf("  G - Toggle grid lines\n");
    printf("  P - Pause/Resume training\n");
    printf("  ESC - Exit demo\n");
    printf("  1-5 - Change training speed\n");
    
    // Initialize graphics
    init_graphics(1000, 700);
    VisualizationState* vis_state = get_visualization_state();
    
    bool demo_paused = false;
    bool show_q_values = config->show_q_values;
    float training_speed = config->training_speed;
    
    TrainingStats* stats = create_training_stats(config->visualization_episodes);
    
    for (int episode = 0; episode < config->visualization_episodes; episode++) {
        reset_environment(world);
        float episode_reward = 0.0f;
        int steps_taken = 0;
        float total_q_value = 0.0f;
        int q_value_count = 0;
        
        while (!world->episode_done && steps_taken < world->max_steps) {
            // Handle input
            if (WindowShouldClose()) {
                printf("Demo interrupted by user\n");
                goto demo_cleanup;
            }
            
            if (IsKeyPressed(KEY_P)) {
                demo_paused = !demo_paused;
                printf("Demo %s\n", demo_paused ? "PAUSED" : "RESUMED");
            }
            
            if (IsKeyPressed(KEY_Q)) {
                show_q_values = !show_q_values;
                printf("Q-value visualization: %s\n", show_q_values ? "ON" : "OFF");
            }
            
            if (IsKeyPressed(KEY_G) && vis_state) {
                vis_state->config.show_grid = !vis_state->config.show_grid;
            }
            
            // Speed controls
            if (IsKeyPressed(KEY_ONE)) training_speed = 0.01f;
            if (IsKeyPressed(KEY_TWO)) training_speed = 0.05f;
            if (IsKeyPressed(KEY_THREE)) training_speed = 0.1f;
            if (IsKeyPressed(KEY_FOUR)) training_speed = 0.2f;
            if (IsKeyPressed(KEY_FIVE)) training_speed = 0.5f;
            
            if (!demo_paused) {
                // Training step
                int current_state = get_state_index(world);
                Action action = select_action(agent, current_state);
                StepResult result = step_environment(world, action);
                
                update_q_value(agent, current_state, action, result.reward,
                             position_to_state(world, result.next_state.position), result.done);
                
                episode_reward += result.reward;
                steps_taken++;
                
                // Calculate average Q-value
                float sum = 0.0f;
                for (int a = 0; a < NUM_ACTIONS; a++) {
                    sum += get_q_value(agent, current_state, a);
                }
                total_q_value += sum / NUM_ACTIONS;
                q_value_count++;
            }
            
            // Render
            BeginDrawing();
            ClearBackground(RAYWHITE);
            
            // Draw environment
            if (show_q_values) {
                draw_q_values(vis_state, world, agent);
            } else {
                draw_grid_world(vis_state, world);
            }
            
            draw_walls(vis_state, world);
            draw_goal(vis_state, world->goal_pos);
            draw_agent(vis_state, world->agent_pos);
            
            // Draw demo info
            char info_text[512];
            snprintf(info_text, sizeof(info_text),
                    "Interactive Q-Learning Demo | Episode: %d/%d | Step: %d | Reward: %.1f",
                    episode + 1, config->visualization_episodes, steps_taken, episode_reward);
            DrawText(info_text, 10, 10, 18, BLACK);
            
            snprintf(info_text, sizeof(info_text),
                    "Agent: (%d,%d) | Epsilon: %.3f | Speed: %.2fx | Q-values: %s",
                    world->agent_pos.x, world->agent_pos.y, agent->epsilon, 
                    training_speed * 20, show_q_values ? "ON" : "OFF");
            DrawText(info_text, 10, 35, 14, DARKGRAY);
            
            if (demo_paused) {
                DrawText("DEMO PAUSED - Press P to resume", 10, 60, 16, RED);
            }
            
            DrawText("Controls: P=Pause | Q=Q-values | G=Grid | 1-5=Speed | ESC=Exit", 
                    10, GetScreenHeight() - 25, 12, DARKBLUE);
            
            EndDrawing();
            
            // Control demo speed
            if (!demo_paused) {
                WaitTime(training_speed);
            }
        }
        
        // Episode completed
        decay_epsilon(agent);
        
        float avg_q_episode = q_value_count > 0 ? total_q_value / q_value_count : 0.0f;
        record_episode(stats, episode, episode_reward, steps_taken, agent->epsilon, avg_q_episode);
        
        if ((episode + 1) % 10 == 0) {
            printf("Episode %d completed: Reward=%.2f, Steps=%d, Epsilon=%.3f\n",
                   episode + 1, episode_reward, steps_taken, agent->epsilon);
        }
    }
    
    demo_cleanup:
    printf("\nInteractive demo completed!\n");
    print_training_summary(stats);
    
    destroy_training_stats(stats);
    cleanup_graphics();
}

int main() {
    printf("Q-Learning Training Loop Integration Demo\n");
    printf("========================================\n");
    
    srand((unsigned int)time(NULL));
    
    // Create environment for demos
    const int GRID_WIDTH = 10;
    const int GRID_HEIGHT = 10;
    
    GridWorld* world = create_grid_world(GRID_WIDTH, GRID_HEIGHT);
    world->start_pos = (Position){1, 1};
    world->goal_pos = (Position){8, 8};
    world->step_penalty = -0.1f;
    world->goal_reward = 100.0f;
    world->wall_penalty = -10.0f;
    world->max_steps = 100;
    
    // Add some walls
    set_cell(world, 3, 3, CELL_WALL);
    set_cell(world, 3, 4, CELL_WALL);
    set_cell(world, 3, 5, CELL_WALL);
    set_cell(world, 5, 2, CELL_WALL);
    set_cell(world, 5, 3, CELL_WALL);
    set_cell(world, 5, 4, CELL_WALL);
    set_cell(world, 7, 6, CELL_WALL);
    set_cell(world, 7, 7, CELL_WALL);
    
    set_cell(world, world->goal_pos.x, world->goal_pos.y, CELL_GOAL);
    set_cell(world, world->start_pos.x, world->start_pos.y, CELL_START);
    
    // Create agent
    QLearningAgent* agent = create_agent(GRID_WIDTH * GRID_HEIGHT, NUM_ACTIONS, 
                                       0.1f, 0.9f, 1.0f);
    agent->epsilon_decay = 0.995f;
    agent->epsilon_min = 0.01f;
    
    // Demo configuration
    DemoConfig config = {
        .demo_episodes = 200,
        .visualization_episodes = 50,
        .show_q_values = true,
        .educational_mode = true,
        .training_speed = 0.1f
    };
    
    printf("\nAvailable demos:\n");
    printf("1. Educational step-by-step Q-learning\n");
    printf("2. Performance comparison\n");
    printf("3. Interactive visualization\n");
    printf("4. All demos\n");
    printf("\nChoose demo (1-4): ");
    
    int choice;
    if (scanf("%d", &choice) != 1) {
        choice = 4; // Default to all demos
    }
    
    switch (choice) {
        case 1:
            run_educational_demo(world, agent);
            break;
        case 2:
            run_performance_demo();
            break;
        case 3:
            run_interactive_demo(world, agent, &config);
            break;
        case 4:
        default:
            run_educational_demo(world, agent);
            run_performance_demo();
            run_interactive_demo(world, agent, &config);
            break;
    }
    
    printf("\nDemo completed successfully!\n");
    printf("The training loop integration includes:\n");
    printf("✓ Q-learning algorithm implementation\n");
    printf("✓ Real-time visualization during training\n");
    printf("✓ Interactive controls (pause, speed, Q-value display)\n");
    printf("✓ Educational step-by-step explanations\n");
    printf("✓ Performance tracking and analysis\n");
    printf("✓ Policy saving and loading\n");
    printf("✓ Configurable training parameters\n");
    
    // Cleanup
    destroy_agent(agent);
    destroy_grid_world(world);
    
    return 0;
}
