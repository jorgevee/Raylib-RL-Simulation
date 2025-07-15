#include "rendering.h"
#include "environment.h"
#include "agent.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(void) {
    // Configuration
    const int SCREEN_WIDTH = 800;
    const int SCREEN_HEIGHT = 600;
    const int CELL_SIZE = 40;
    const int GRID_WIDTH = 10;
    const int GRID_HEIGHT = 10;
    
    printf("Starting RL Agent Visualization Demo\n");
    
    // Initialize graphics
    init_graphics(SCREEN_WIDTH, SCREEN_HEIGHT);
    
    // Create environment
    GridWorld* world = create_grid_world(GRID_WIDTH, GRID_HEIGHT);
    if (!world) {
        printf("Error: Failed to create grid world\n");
        cleanup_graphics();
        return -1;
    }
    
    // Set up a simple environment
    world->start_pos = (Position){1, 1};
    world->goal_pos = (Position){8, 8};
    world->agent_pos = world->start_pos;
    
    // Add some walls for testing
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
    QLearningAgent* agent = create_agent(num_states, NUM_ACTIONS, 0.1f, 0.9f, 0.1f);
    if (!agent) {
        printf("Error: Failed to create agent\n");
        destroy_grid_world(world);
        cleanup_graphics();
        return -1;
    }
    
    // Initialize some sample Q-values for visualization
    printf("Initializing sample Q-values...\n");
    for (int state = 0; state < num_states; state++) {
        for (int action = 0; action < NUM_ACTIONS; action++) {
            // Create some interesting Q-value patterns
            Position pos = state_to_position(world, state);
            
            // Higher values closer to goal
            float distance_to_goal = sqrt((pos.x - world->goal_pos.x) * (pos.x - world->goal_pos.x) + 
                                        (pos.y - world->goal_pos.y) * (pos.y - world->goal_pos.y));
            
            // Base Q-value inversely related to distance
            float base_q = 10.0f - distance_to_goal;
            
            // Add some action-specific variations
            float action_bonus = 0.0f;
            switch (action) {
                case ACTION_UP:
                    action_bonus = (pos.y > world->goal_pos.y) ? 2.0f : -1.0f;
                    break;
                case ACTION_DOWN:
                    action_bonus = (pos.y < world->goal_pos.y) ? 2.0f : -1.0f;
                    break;
                case ACTION_LEFT:
                    action_bonus = (pos.x > world->goal_pos.x) ? 2.0f : -1.0f;
                    break;
                case ACTION_RIGHT:
                    action_bonus = (pos.x < world->goal_pos.x) ? 2.0f : -1.0f;
                    break;
            }
            
            set_q_value(agent, state, action, base_q + action_bonus + (rand() % 100) / 100.0f);
        }
    }
    
    printf("Starting visualization loop...\n");
    printf("Controls:\n");
    printf("  Q - Toggle Q-value visualization\n");
    printf("  G - Toggle grid lines\n");
    printf("  ESC - Exit\n");
    printf("  Arrow Keys - Move agent\n");
    
    // Get visualization state for configuration
    VisualizationState* vis_state = get_visualization_state();
    
    // Main game loop
    while (!WindowShouldClose()) {
        // Handle input
        if (IsKeyPressed(KEY_Q)) {
            if (vis_state) {
                vis_state->config.show_q_values = !vis_state->config.show_q_values;
                printf("Q-value visualization: %s\n", 
                       vis_state->config.show_q_values ? "ON" : "OFF");
            }
        }
        
        if (IsKeyPressed(KEY_G)) {
            if (vis_state) {
                vis_state->config.show_grid = !vis_state->config.show_grid;
                printf("Grid lines: %s\n", 
                       vis_state->config.show_grid ? "ON" : "OFF");
            }
        }
        
        // Handle agent movement
        Position new_pos = world->agent_pos;
        bool moved = false;
        
        if (IsKeyPressed(KEY_UP) && new_pos.y > 0) {
            new_pos.y--;
            moved = true;
        } else if (IsKeyPressed(KEY_DOWN) && new_pos.y < world->height - 1) {
            new_pos.y++;
            moved = true;
        } else if (IsKeyPressed(KEY_LEFT) && new_pos.x > 0) {
            new_pos.x--;
            moved = true;
        } else if (IsKeyPressed(KEY_RIGHT) && new_pos.x < world->width - 1) {
            new_pos.x++;
            moved = true;
        }
        
        // Check if new position is valid (not a wall)
        if (moved && is_walkable(world, new_pos.x, new_pos.y)) {
            world->agent_pos = new_pos;
            
            // Check if reached goal
            if (positions_equal(world->agent_pos, world->goal_pos)) {
                printf("Goal reached! Resetting agent to start position.\n");
                world->agent_pos = world->start_pos;
            }
        }
        
        // Rendering
        BeginDrawing();
        
        // Draw Q-values first (as background)
        if (vis_state && vis_state->config.show_q_values) {
            draw_q_values(vis_state, world, agent);
        } else {
            // Draw normal grid world
            draw_grid_world(vis_state, world);
        }
        
        // Draw walls on top
        draw_walls(vis_state, world);
        
        // Draw goal
        draw_goal(vis_state, world->goal_pos);
        
        // Draw agent on top of everything
        draw_agent(vis_state, world->agent_pos);
        
        // Draw UI text
        DrawText("RL Agent Visualization Demo", 10, SCREEN_HEIGHT - 90, 20, BLACK);
        DrawText("Q: Toggle Q-values | G: Toggle Grid | Arrows: Move Agent", 10, SCREEN_HEIGHT - 60, 16, DARKGRAY);
        
        char info_text[256];
        snprintf(info_text, sizeof(info_text), 
                "Agent: (%d,%d) | Goal: (%d,%d) | Q-values: %s", 
                world->agent_pos.x, world->agent_pos.y,
                world->goal_pos.x, world->goal_pos.y,
                (vis_state && vis_state->config.show_q_values) ? "ON" : "OFF");
        DrawText(info_text, 10, SCREEN_HEIGHT - 30, 16, DARKBLUE);
        
        EndDrawing();
    }
    
    printf("Cleaning up...\n");
    
    // Cleanup
    destroy_agent(agent);
    destroy_grid_world(world);
    cleanup_graphics();
    
    printf("Demo completed successfully!\n");
    return 0;
}
