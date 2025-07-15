#ifndef RENDERING_H
#define RENDERING_H

#include <stdbool.h>
#include "raylib.h"
#include "environment.h"
#include "agent.h"

// Rendering configuration structure
typedef struct {
    int cell_size;              // Size of each grid cell in pixels
    int screen_width;           // Screen width
    int screen_height;          // Screen height
    bool show_q_values;         // Whether to display Q-values
    bool show_grid;             // Whether to show grid lines
    bool show_agent_trail;      // Whether to show agent's path
    bool show_statistics;       // Whether to show training stats
    float animation_speed;      // Speed of animations (0.0-1.0)
    int fps_target;             // Target FPS
    bool vsync_enabled;         // Whether VSync is enabled
} RenderConfig;

// Color scheme for visualization
typedef struct {
    Color empty_cell;           // Color for empty cells
    Color wall_cell;            // Color for walls
    Color goal_cell;            // Color for goal
    Color agent_color;          // Color for agent
    Color obstacle_color;       // Color for obstacles
    Color start_cell;           // Color for start position
    Color grid_lines;           // Color for grid lines
    Color text_color;           // Color for text
    Color background;           // Background color
    Color q_value_positive;     // Color for positive Q-values
    Color q_value_negative;     // Color for negative Q-values
    Color trail_color;          // Color for agent trail
} ColorScheme;

// Agent trail for visualization
typedef struct {
    Position* positions;        // Array of positions
    int count;                  // Current number of positions
    int capacity;               // Maximum capacity
    int head;                   // Head index for circular buffer
    float* timestamps;          // Timestamps for fade effect
} AgentTrail;

// Animation state for smooth movement
typedef struct {
    Vector2f current_pos;       // Current animated position
    Vector2f target_pos;        // Target position
    float animation_time;       // Current animation time
    float animation_duration;   // Duration of animation
    bool is_animating;          // Whether animation is active
} AnimationState;

// UI element positions and sizes
typedef struct {
    Rectangle stats_panel;      // Statistics panel area
    Rectangle control_panel;    // Control buttons area
    Rectangle q_value_panel;    // Q-value display area
    Rectangle grid_area;        // Main grid display area
    int margin;                 // Margin around elements
    int panel_height;           // Height of info panels
} UILayout;

// Text rendering information
typedef struct {
    Font font;                  // Font for text rendering
    int font_size;              // Size of font
    int line_spacing;           // Spacing between lines
    bool font_loaded;           // Whether custom font is loaded
} TextRenderer;

// Visualization state
typedef struct {
    RenderConfig config;        // Rendering configuration
    ColorScheme colors;         // Color scheme
    AgentTrail trail;           // Agent movement trail
    AnimationState animation;   // Animation state
    UILayout layout;            // UI layout information
    TextRenderer text;          // Text rendering
    Camera2D camera;            // 2D camera for zooming/panning
    bool camera_enabled;        // Whether camera controls are active
} VisualizationState;

// Function declarations for rendering system
VisualizationState* init_visualization(int screen_width, int screen_height, int cell_size);
void destroy_visualization(VisualizationState* vis);
void update_visualization(VisualizationState* vis, GridWorld* world, QLearningAgent* agent, TrainingStats* stats);
void render_frame(VisualizationState* vis, GridWorld* world, QLearningAgent* agent, TrainingStats* stats);

// Basic graphics initialization and cleanup
void init_graphics(int screen_width, int screen_height);
void cleanup_graphics(void);
VisualizationState* get_visualization_state(void);

// Grid rendering functions
void draw_grid_world(VisualizationState* vis, GridWorld* world);
void draw_grid_lines(VisualizationState* vis, GridWorld* world);
void draw_cell(VisualizationState* vis, int x, int y, CellType type);
void draw_agent(VisualizationState* vis, Position pos);
void draw_goal(VisualizationState* vis, Position pos);
void draw_walls(VisualizationState* vis, GridWorld* world);

// Q-value visualization
void draw_q_values(VisualizationState* vis, GridWorld* world, QLearningAgent* agent);
void draw_q_value_arrows(VisualizationState* vis, GridWorld* world, QLearningAgent* agent);
void draw_q_value_heatmap(VisualizationState* vis, GridWorld* world, QLearningAgent* agent);
void draw_policy_arrows(VisualizationState* vis, GridWorld* world, QLearningAgent* agent);

// Trail and animation functions
AgentTrail* create_agent_trail(int capacity);
void destroy_agent_trail(AgentTrail* trail);
void add_trail_position(AgentTrail* trail, Position pos);
void draw_agent_trail(VisualizationState* vis, AgentTrail* trail);
void update_animation(VisualizationState* vis, Position target);
void draw_animated_agent(VisualizationState* vis);

// UI and statistics rendering
void draw_statistics_panel(VisualizationState* vis, TrainingStats* stats);
void draw_control_panel(VisualizationState* vis);
void draw_episode_info(VisualizationState* vis, GridWorld* world, int episode);
void draw_agent_info(VisualizationState* vis, QLearningAgent* agent);
void draw_performance_graph(VisualizationState* vis, TrainingStats* stats);

// Color and utility functions
ColorScheme create_default_color_scheme(void);
ColorScheme create_dark_color_scheme(void);
Color lerp_color(Color a, Color b, float t);
Color q_value_to_color(float q_value, float min_q, float max_q);
Rectangle get_cell_rect(VisualizationState* vis, int x, int y);
Vector2 grid_to_screen(VisualizationState* vis, Position pos);
Position screen_to_grid(VisualizationState* vis, Vector2 screen_pos);

// Input handling
void handle_mouse_input(VisualizationState* vis, GridWorld* world);
void handle_keyboard_input(VisualizationState* vis, GridWorld* world, QLearningAgent* agent);
bool is_cell_clicked(VisualizationState* vis, int x, int y);

// Configuration and settings
void set_render_config(VisualizationState* vis, RenderConfig config);
void toggle_visualization_option(VisualizationState* vis, const char* option);
void save_screenshot(VisualizationState* vis, const char* filename);
void export_visualization_config(VisualizationState* vis, const char* filename);
void load_visualization_config(VisualizationState* vis, const char* filename);

// Debug and development functions
void draw_debug_info(VisualizationState* vis, GridWorld* world, QLearningAgent* agent);
void draw_fps_counter(VisualizationState* vis);
void draw_memory_usage(VisualizationState* vis);

#endif // RENDERING_H
