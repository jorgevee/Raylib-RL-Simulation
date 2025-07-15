#include "rendering.h"
#include "raylib.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Global visualization state - simplified for core functions
static VisualizationState* g_vis_state = NULL;

// Helper function to create default color scheme
ColorScheme create_default_color_scheme(void) {
    ColorScheme colors = {
        .empty_cell = LIGHTGRAY,
        .wall_cell = DARKGRAY,
        .goal_cell = GREEN,
        .agent_color = BLUE,
        .obstacle_color = RED,
        .start_cell = YELLOW,
        .grid_lines = GRAY,
        .text_color = BLACK,
        .background = WHITE,
        .q_value_positive = LIME,
        .q_value_negative = PINK,
        .trail_color = SKYBLUE
    };
    return colors;
}

// Initialize graphics system
void init_graphics(int screen_width, int screen_height) {
    // Initialize raylib window
    InitWindow(screen_width, screen_height, "RL Agent Visualization");
    SetTargetFPS(60);
    
    // Initialize global visualization state if not already done
    if (g_vis_state == NULL) {
        g_vis_state = (VisualizationState*)malloc(sizeof(VisualizationState));
        if (g_vis_state == NULL) {
            printf("Error: Failed to allocate visualization state\n");
            return;
        }
        
        // Initialize basic configuration
        g_vis_state->config.screen_width = screen_width;
        g_vis_state->config.screen_height = screen_height;
        g_vis_state->config.cell_size = 40; // Default cell size
        g_vis_state->config.show_q_values = true;
        g_vis_state->config.show_grid = true;
        g_vis_state->config.fps_target = 60;
        
        // Initialize color scheme
        g_vis_state->colors = create_default_color_scheme();
        
        // Initialize layout
        g_vis_state->layout.margin = 10;
        g_vis_state->layout.panel_height = 100;
        g_vis_state->layout.grid_area = (Rectangle){
            10, 10, 
            screen_width - 20, 
            screen_height - 120
        };
        
        // Initialize text renderer
        g_vis_state->text.font = GetFontDefault();
        g_vis_state->text.font_size = 20;
        g_vis_state->text.line_spacing = 5;
        g_vis_state->text.font_loaded = false;
    }
    
    printf("Graphics initialized: %dx%d\n", screen_width, screen_height);
}

// Get cell rectangle for drawing
Rectangle get_cell_rect(VisualizationState* vis, int x, int y) {
    if (!vis) return (Rectangle){0, 0, 0, 0};
    
    float cell_size = (float)vis->config.cell_size;
    float start_x = vis->layout.grid_area.x;
    float start_y = vis->layout.grid_area.y;
    
    return (Rectangle){
        start_x + x * cell_size,
        start_y + y * cell_size,
        cell_size,
        cell_size
    };
}

// Draw a single cell
void draw_cell(VisualizationState* vis, int x, int y, CellType type) {
    if (!vis) return;
    
    Rectangle cell_rect = get_cell_rect(vis, x, y);
    Color cell_color;
    
    switch (type) {
        case CELL_EMPTY:
            cell_color = vis->colors.empty_cell;
            break;
        case CELL_WALL:
            cell_color = vis->colors.wall_cell;
            break;
        case CELL_GOAL:
            cell_color = vis->colors.goal_cell;
            break;
        case CELL_AGENT:
            cell_color = vis->colors.agent_color;
            break;
        case CELL_OBSTACLE:
            cell_color = vis->colors.obstacle_color;
            break;
        case CELL_START:
            cell_color = vis->colors.start_cell;
            break;
        default:
            cell_color = vis->colors.empty_cell;
            break;
    }
    
    DrawRectangleRec(cell_rect, cell_color);
    
    // Draw cell border if grid is enabled
    if (vis->config.show_grid) {
        DrawRectangleLinesEx(cell_rect, 1.0f, vis->colors.grid_lines);
    }
}

// Draw the entire grid world
void draw_grid_world(VisualizationState* vis, GridWorld* world) {
    if (!world || !vis) return;
    
    // Clear background
    ClearBackground(vis->colors.background);
    
    // Draw all cells
    for (int y = 0; y < world->height; y++) {
        for (int x = 0; x < world->width; x++) {
            CellType cell_type = world->grid[y][x];
            draw_cell(vis, x, y, cell_type);
        }
    }
    
    // Draw grid lines if enabled
    if (vis->config.show_grid) {
        draw_grid_lines(vis, world);
    }
}

// Draw grid lines
void draw_grid_lines(VisualizationState* vis, GridWorld* world) {
    if (!vis || !world) return;
    
    float cell_size = (float)vis->config.cell_size;
    float start_x = vis->layout.grid_area.x;
    float start_y = vis->layout.grid_area.y;
    float grid_width = world->width * cell_size;
    float grid_height = world->height * cell_size;
    
    // Draw vertical lines
    for (int x = 0; x <= world->width; x++) {
        float line_x = start_x + x * cell_size;
        DrawLine((int)line_x, (int)start_y, 
                (int)line_x, (int)(start_y + grid_height), 
                vis->colors.grid_lines);
    }
    
    // Draw horizontal lines
    for (int y = 0; y <= world->height; y++) {
        float line_y = start_y + y * cell_size;
        DrawLine((int)start_x, (int)line_y, 
                (int)(start_x + grid_width), (int)line_y, 
                vis->colors.grid_lines);
    }
}

// Draw agent at specified position
void draw_agent(VisualizationState* vis, Position pos) {
    if (!vis) return;
    
    Rectangle cell_rect = get_cell_rect(vis, pos.x, pos.y);
    
    // Draw agent as a circle in the center of the cell
    float center_x = cell_rect.x + cell_rect.width / 2;
    float center_y = cell_rect.y + cell_rect.height / 2;
    float radius = (vis->config.cell_size * 0.3f); // Agent is 30% of cell size
    
    DrawCircle((int)center_x, (int)center_y, radius, vis->colors.agent_color);
    
    // Draw agent border
    DrawCircleLines((int)center_x, (int)center_y, radius, BLACK);
}

// Draw goal at specified position
void draw_goal(VisualizationState* vis, Position pos) {
    if (!vis) return;
    
    Rectangle cell_rect = get_cell_rect(vis, pos.x, pos.y);
    
    // Draw goal as a filled rectangle with special pattern
    DrawRectangleRec(cell_rect, vis->colors.goal_cell);
    
    // Draw goal symbol (star-like pattern)
    float center_x = cell_rect.x + cell_rect.width / 2;
    float center_y = cell_rect.y + cell_rect.height / 2;
    float size = vis->config.cell_size * 0.4f;
    
    // Draw cross pattern for goal
    DrawLine((int)(center_x - size/2), (int)center_y, 
            (int)(center_x + size/2), (int)center_y, DARKGREEN);
    DrawLine((int)center_x, (int)(center_y - size/2), 
            (int)center_x, (int)(center_y + size/2), DARKGREEN);
}

// Draw walls in the grid world
void draw_walls(VisualizationState* vis, GridWorld* world) {
    if (!world || !vis) return;
    
    for (int y = 0; y < world->height; y++) {
        for (int x = 0; x < world->width; x++) {
            if (world->grid[y][x] == CELL_WALL) {
                Rectangle cell_rect = get_cell_rect(vis, x, y);
                DrawRectangleRec(cell_rect, vis->colors.wall_cell);
                
                // Add some texture to walls
                DrawRectangleLinesEx(cell_rect, 2.0f, BLACK);
            }
        }
    }
}

// Convert Q-value to color for visualization
Color q_value_to_color(float q_value, float min_q, float max_q) {
    if (!g_vis_state) return WHITE;
    
    if (max_q == min_q) return g_vis_state->colors.empty_cell;
    
    // Normalize Q-value to 0-1 range
    float normalized = (q_value - min_q) / (max_q - min_q);
    
    // Interpolate between negative and positive colors
    if (normalized < 0.5f) {
        // Interpolate from negative to neutral
        float t = normalized * 2.0f;
        return (Color){
            (unsigned char)(g_vis_state->colors.q_value_negative.r * (1.0f - t) + 128 * t),
            (unsigned char)(g_vis_state->colors.q_value_negative.g * (1.0f - t) + 128 * t),
            (unsigned char)(g_vis_state->colors.q_value_negative.b * (1.0f - t) + 128 * t),
            180
        };
    } else {
        // Interpolate from neutral to positive
        float t = (normalized - 0.5f) * 2.0f;
        return (Color){
            (unsigned char)(128 * (1.0f - t) + g_vis_state->colors.q_value_positive.r * t),
            (unsigned char)(128 * (1.0f - t) + g_vis_state->colors.q_value_positive.g * t),
            (unsigned char)(128 * (1.0f - t) + g_vis_state->colors.q_value_positive.b * t),
            180
        };
    }
}

// Draw Q-values visualization
void draw_q_values(VisualizationState* vis, GridWorld* world, QLearningAgent* agent) {
    if (!agent || !world || !vis || !vis->config.show_q_values) return;
    
    // Find min and max Q-values for normalization
    float min_q = 999999.0f, max_q = -999999.0f;
    
    for (int state = 0; state < agent->num_states; state++) {
        for (int action = 0; action < agent->num_actions; action++) {
            float q_val = agent->q_table[state][action];
            if (q_val < min_q) min_q = q_val;
            if (q_val > max_q) max_q = q_val;
        }
    }
    
    // Draw Q-value heatmap
    for (int y = 0; y < world->height; y++) {
        for (int x = 0; x < world->width; x++) {
            if (world->grid[y][x] != CELL_WALL) {
                int state = y * world->width + x;
                
                if (state < agent->num_states) {
                    // Find maximum Q-value for this state
                    float max_q_state = -999999.0f;
                    Action best_action = ACTION_UP;
                    
                    for (int action = 0; action < agent->num_actions; action++) {
                        float q_val = agent->q_table[state][action];
                        if (q_val > max_q_state) {
                            max_q_state = q_val;
                            best_action = action;
                        }
                    }
                    
                    // Draw Q-value as background color
                    Rectangle cell_rect = get_cell_rect(vis, x, y);
                    Color q_color = q_value_to_color(max_q_state, min_q, max_q);
                    DrawRectangleRec(cell_rect, q_color);
                    
                    // Draw policy arrow showing best action
                    float center_x = cell_rect.x + cell_rect.width / 2;
                    float center_y = cell_rect.y + cell_rect.height / 2;
                    float arrow_size = vis->config.cell_size * 0.3f;
                    
                    Vector2 start = {center_x, center_y};
                    Vector2 end = start;
                    
                    switch (best_action) {
                        case ACTION_UP:
                            end.y -= arrow_size;
                            break;
                        case ACTION_DOWN:
                            end.y += arrow_size;
                            break;
                        case ACTION_LEFT:
                            end.x -= arrow_size;
                            break;
                        case ACTION_RIGHT:
                            end.x += arrow_size;
                            break;
                    }
                    
                    // Draw arrow
                    DrawLineEx(start, end, 3.0f, BLACK);
                    
                    // Draw arrowhead
                    Vector2 direction = {end.x - start.x, end.y - start.y};
                    float length = sqrtf(direction.x * direction.x + direction.y * direction.y);
                    if (length > 0) {
                        direction.x /= length;
                        direction.y /= length;
                        
                        Vector2 arrowhead1 = {
                            end.x - direction.x * 8 + direction.y * 4,
                            end.y - direction.y * 8 - direction.x * 4
                        };
                        Vector2 arrowhead2 = {
                            end.x - direction.x * 8 - direction.y * 4,
                            end.y - direction.y * 8 + direction.x * 4
                        };
                        
                        DrawLineEx(end, arrowhead1, 2.0f, BLACK);
                        DrawLineEx(end, arrowhead2, 2.0f, BLACK);
                    }
                    
                    // Draw Q-value text if cell is large enough
                    if (vis->config.cell_size > 60) {
                        char q_text[16];
                        snprintf(q_text, sizeof(q_text), "%.2f", max_q_state);
                        int text_width = MeasureText(q_text, 12);
                        DrawText(q_text, 
                                (int)(center_x - text_width/2), 
                                (int)(center_y + arrow_size/2 + 5), 
                                12, BLACK);
                    }
                }
            }
        }
    }
}

// Get global visualization state (for external access)
VisualizationState* get_visualization_state(void) {
    return g_vis_state;
}

// Cleanup function
void cleanup_graphics(void) {
    if (g_vis_state) {
        free(g_vis_state);
        g_vis_state = NULL;
    }
    CloseWindow();
}

// Simple initialization function that matches the header's advanced structure
VisualizationState* init_visualization(int screen_width, int screen_height, int cell_size) {
    init_graphics(screen_width, screen_height);
    if (g_vis_state) {
        g_vis_state->config.cell_size = cell_size;
    }
    return g_vis_state;
}

// Destroy visualization (matches header)
void destroy_visualization(VisualizationState* vis) {
    cleanup_graphics();
}
