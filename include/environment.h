#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include <stdbool.h>
#include "agent.h"

// Basic position structure
typedef struct {
    int x, y;
} Position;

// Floating point vector for smooth animations
typedef struct {
    float x, y;
} Vector2f;

// Cell types in the grid world
typedef enum {
    CELL_EMPTY = 0,
    CELL_WALL = 1,
    CELL_GOAL = 2,
    CELL_AGENT = 3,
    CELL_OBSTACLE = 4,
    CELL_START = 5
} CellType;

// Grid world environment structure
typedef struct {
    int width, height;          // Grid dimensions
    CellType** grid;           // 2D grid of cell types
    Position agent_pos;        // Current agent position
    Position goal_pos;         // Goal position
    Position start_pos;        // Starting position
    int episode_steps;         // Steps taken in current episode
    bool episode_done;         // Whether episode is finished
    float total_reward;        // Total reward accumulated this episode
    int max_steps;             // Maximum steps per episode
    float step_penalty;        // Penalty for each step (-1.0 typical)
    float goal_reward;         // Reward for reaching goal (+100.0 typical)
    float wall_penalty;        // Penalty for hitting wall (-10.0 typical)
} GridWorld;

// Environment configuration
typedef struct {
    int width, height;
    float step_penalty;
    float goal_reward;
    float wall_penalty;
    int max_steps;
    bool stochastic;           // Whether actions have randomness
    float action_noise;        // Probability of random action (0.0-1.0)
} EnvironmentConfig;

// State representation for the agent
typedef struct {
    int state_index;           // 1D state representation
    Position position;         // 2D position representation
    bool is_terminal;          // Whether this is a terminal state
    bool is_valid;             // Whether this state is valid
} State;

// Step result structure
typedef struct {
    State next_state;          // Resulting state after action
    float reward;              // Immediate reward
    bool done;                 // Whether episode is terminated
    bool valid_action;         // Whether the action was valid
} StepResult;

// Function declarations for environment management
GridWorld* create_grid_world(int width, int height);
GridWorld* create_grid_world_from_config(EnvironmentConfig config);
void destroy_grid_world(GridWorld* world);
void reset_environment(GridWorld* world);
StepResult step_environment(GridWorld* world, Action action);
int step(GridWorld* world, Action action, float* reward);
int get_state_index(GridWorld* world);

// Grid manipulation functions
void set_cell(GridWorld* world, int x, int y, CellType type);
CellType get_cell(GridWorld* world, int x, int y);
bool is_valid_position(GridWorld* world, int x, int y);
bool is_walkable(GridWorld* world, int x, int y);

// State conversion functions
int position_to_state(GridWorld* world, Position pos);
Position state_to_position(GridWorld* world, int state);
State get_current_state(GridWorld* world);
bool is_terminal_state(GridWorld* world, Position pos);

// Environment setup functions
void set_random_goal(GridWorld* world);
void set_random_start(GridWorld* world);
void add_random_walls(GridWorld* world, float wall_density);
void clear_environment(GridWorld* world);

// Maze generation functions
void generate_simple_maze(GridWorld* world);
void generate_random_maze(GridWorld* world, float complexity);
void create_corridor_maze(GridWorld* world);

// Utility functions
float calculate_manhattan_distance(Position a, Position b);
float calculate_euclidean_distance(Position a, Position b);
bool positions_equal(Position a, Position b);
Position get_new_position(Position current, Action action);

// Reward calculation
float calculate_reward(GridWorld* world, Position old_pos, Position new_pos, bool valid_move);
float get_state_reward(GridWorld* world, Position pos);

// Reward configuration and validation
bool validate_reward_values(GridWorld* world);
bool set_reward_values(GridWorld* world, float goal_reward, float wall_penalty, float step_penalty);
void get_reward_values(GridWorld* world, float* goal_reward, float* wall_penalty, float* step_penalty);

// Environment validation
bool validate_environment(GridWorld* world);
void print_environment_info(GridWorld* world);

#endif // ENVIRONMENT_H
