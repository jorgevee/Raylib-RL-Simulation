#include "../include/environment.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// Create a new grid world environment
GridWorld* create_grid_world(int width, int height) {
    // Validate input parameters
    if (width <= 0 || height <= 0) {
        fprintf(stderr, "Error: Grid dimensions must be positive (width=%d, height=%d)\n", width, height);
        return NULL;
    }
    
    // Allocate memory for the GridWorld structure
    GridWorld* world = (GridWorld*)malloc(sizeof(GridWorld));
    if (!world) {
        fprintf(stderr, "Error: Failed to allocate memory for GridWorld structure\n");
        return NULL;
    }
    
    // Initialize basic dimensions
    world->width = width;
    world->height = height;
    
    // Allocate memory for the 2D grid
    world->grid = (CellType**)malloc(height * sizeof(CellType*));
    if (!world->grid) {
        fprintf(stderr, "Error: Failed to allocate memory for grid rows\n");
        free(world);
        return NULL;
    }
    
    // Allocate memory for each row and initialize to empty
    for (int y = 0; y < height; y++) {
        world->grid[y] = (CellType*)malloc(width * sizeof(CellType));
        if (!world->grid[y]) {
            fprintf(stderr, "Error: Failed to allocate memory for grid row %d\n", y);
            // Clean up previously allocated rows
            for (int cleanup_y = 0; cleanup_y < y; cleanup_y++) {
                free(world->grid[cleanup_y]);
            }
            free(world->grid);
            free(world);
            return NULL;
        }
        
        // Initialize all cells to empty
        for (int x = 0; x < width; x++) {
            world->grid[y][x] = CELL_EMPTY;
        }
    }
    
    // Initialize positions (default: agent at top-left, goal at bottom-right)
    world->agent_pos.x = 0;
    world->agent_pos.y = 0;
    world->start_pos.x = 0;
    world->start_pos.y = 0;
    world->goal_pos.x = width - 1;
    world->goal_pos.y = height - 1;
    
    // Mark the start and goal positions in the grid
    world->grid[world->start_pos.y][world->start_pos.x] = CELL_START;
    world->grid[world->goal_pos.y][world->goal_pos.x] = CELL_GOAL;
    
    // Initialize episode tracking variables
    world->episode_steps = 0;
    world->episode_done = false;
    world->total_reward = 0.0f;
    
    // Set default configuration values
    world->max_steps = width * height * 2;  // Reasonable upper bound
    world->step_penalty = -1.0f;            // Small penalty for each step
    world->goal_reward = 100.0f;            // Large reward for reaching goal
    world->wall_penalty = -10.0f;           // Penalty for hitting walls
    
    printf("Created grid world: %dx%d, agent at (%d,%d), goal at (%d,%d)\n", 
           width, height, world->agent_pos.x, world->agent_pos.y, 
           world->goal_pos.x, world->goal_pos.y);
    
    return world;
}

// Create a grid world from configuration
GridWorld* create_grid_world_from_config(EnvironmentConfig config) {
    // Validate configuration parameters
    if (config.width <= 0 || config.height <= 0) {
        fprintf(stderr, "Error: Grid dimensions must be positive (width=%d, height=%d)\n", 
                config.width, config.height);
        return NULL;
    }
    
    if (config.max_steps <= 0) {
        fprintf(stderr, "Error: max_steps must be positive (max_steps=%d)\n", config.max_steps);
        return NULL;
    }
    
    // Create basic grid world
    GridWorld* world = create_grid_world(config.width, config.height);
    if (!world) {
        return NULL;
    }
    
    // Apply configuration values
    world->step_penalty = config.step_penalty;
    world->goal_reward = config.goal_reward;
    world->wall_penalty = config.wall_penalty;
    world->max_steps = config.max_steps;
    
    // Validate reward values
    if (!validate_reward_values(world)) {
        fprintf(stderr, "Warning: Reward values may not promote optimal learning\n");
    }
    
    printf("Created configured grid world: %dx%d, rewards: goal=%.1f, wall=%.1f, step=%.1f\n",
           world->width, world->height, world->goal_reward, world->wall_penalty, world->step_penalty);
    
    return world;
}

// Reset the environment to its initial state
void reset_environment(GridWorld* world) {
    if (!world) {
        fprintf(stderr, "Error: Cannot reset NULL GridWorld\n");
        return;
    }
    
    // Reset agent to starting position
    world->agent_pos.x = world->start_pos.x;
    world->agent_pos.y = world->start_pos.y;
    
    // Reset episode tracking variables
    world->episode_steps = 0;
    world->episode_done = false;
    world->total_reward = 0.0f;
    
    printf("Environment reset: agent at (%d,%d), episode ready\n", 
           world->agent_pos.x, world->agent_pos.y);
}

// Convert 2D position to 1D state index
int get_state_index(GridWorld* world) {
    if (!world) {
        fprintf(stderr, "Error: Cannot get state index from NULL GridWorld\n");
        return -1;
    }
    
    return world->agent_pos.y * world->width + world->agent_pos.x;
}

// Check if a position is terminal (matches header signature)
bool is_terminal_state(GridWorld* world, Position pos) {
    if (!world) {
        fprintf(stderr, "Error: Cannot check terminal state of NULL GridWorld\n");
        return true; // Safer to assume terminal if invalid
    }
    
    // Terminal if position reached the goal
    return positions_equal(pos, world->goal_pos);
}

// Execute an action and return the next state
int step(GridWorld* world, Action action, float* reward) {
    if (!world || !reward) {
        fprintf(stderr, "Error: Invalid parameters for step function\n");
        return -1; // Error code
    }
    
    if (world->episode_done) {
        fprintf(stderr, "Warning: Episode already completed, reset environment first\n");
        *reward = 0.0f;
        return get_state_index(world);
    }
    
    // Save current position
    Position old_pos = world->agent_pos;
    Position new_pos = old_pos;
    
    // Calculate new position based on action
    switch (action) {
        case ACTION_UP:
            new_pos.y = old_pos.y - 1;
            break;
        case ACTION_DOWN:
            new_pos.y = old_pos.y + 1;
            break;
        case ACTION_LEFT:
            new_pos.x = old_pos.x - 1;
            break;
        case ACTION_RIGHT:
            new_pos.x = old_pos.x + 1;
            break;
        default:
            fprintf(stderr, "Warning: Invalid action %d\n", action);
            *reward = 0.0f;
            return get_state_index(world);
    }
    
    // Check if new position is valid and walkable
    bool valid_move = is_valid_position(world, new_pos.x, new_pos.y) && 
                      is_walkable(world, new_pos.x, new_pos.y);
    
    if (valid_move) {
        // Move agent to new position
        world->agent_pos = new_pos;
    }
    
    // Calculate reward
    *reward = calculate_reward(world, old_pos, world->agent_pos, valid_move);
    world->total_reward += *reward;
    
    // Increment step counter
    world->episode_steps++;
    
    // Check if episode is done
    world->episode_done = is_terminal_state(world, world->agent_pos) || 
                          (world->episode_steps >= world->max_steps);
    
    return get_state_index(world);
}

// Step environment with structured result (alternative interface)
StepResult step_environment(GridWorld* world, Action action) {
    StepResult result = {0};
    
    if (!world) {
        fprintf(stderr, "Error: Cannot step NULL GridWorld\n");
        result.next_state.state_index = -1;
        result.next_state.is_valid = false;
        result.reward = 0.0f;
        result.done = true;
        result.valid_action = false;
        return result;
    }
    
    if (world->episode_done) {
        fprintf(stderr, "Warning: Episode already completed in step_environment\n");
        result.next_state = get_current_state(world);
        result.reward = 0.0f;
        result.done = true;
        result.valid_action = false;
        return result;
    }
    
    // Save current position
    Position old_pos = world->agent_pos;
    Position new_pos = old_pos;
    
    // Calculate new position based on action
    switch (action) {
        case ACTION_UP:
            new_pos.y = old_pos.y - 1;
            break;
        case ACTION_DOWN:
            new_pos.y = old_pos.y + 1;
            break;
        case ACTION_LEFT:
            new_pos.x = old_pos.x - 1;
            break;
        case ACTION_RIGHT:
            new_pos.x = old_pos.x + 1;
            break;
        default:
            fprintf(stderr, "Warning: Invalid action %d in step_environment\n", action);
            result.next_state = get_current_state(world);
            result.reward = 0.0f;
            result.done = world->episode_done;
            result.valid_action = false;
            return result;
    }
    
    // Check if new position is valid and walkable
    bool valid_move = is_valid_position(world, new_pos.x, new_pos.y) && 
                      is_walkable(world, new_pos.x, new_pos.y);
    
    if (valid_move) {
        // Move agent to new position
        world->agent_pos = new_pos;
    }
    
    // Calculate reward
    result.reward = calculate_reward(world, old_pos, world->agent_pos, valid_move);
    world->total_reward += result.reward;
    
    // Increment step counter
    world->episode_steps++;
    
    // Check if episode is done
    world->episode_done = is_terminal_state(world, world->agent_pos) || 
                          (world->episode_steps >= world->max_steps);
    
    // Fill result structure
    result.next_state = get_current_state(world);
    result.done = world->episode_done;
    result.valid_action = valid_move;
    
    return result;
}

// Destroy and free all memory associated with the grid world
void destroy_grid_world(GridWorld* world) {
    if (!world) {
        return; // Nothing to destroy
    }
    
    // Free the 2D grid
    if (world->grid) {
        for (int y = 0; y < world->height; y++) {
            if (world->grid[y]) {
                free(world->grid[y]);
            }
        }
        free(world->grid);
    }
    
    // Free the main structure
    free(world);
    
    printf("GridWorld destroyed and memory freed\n");
}

// Helper function implementations that are referenced above

// Check if a position is valid (within bounds)
bool is_valid_position(GridWorld* world, int x, int y) {
    if (!world) return false;
    return (x >= 0 && x < world->width && y >= 0 && y < world->height);
}

// Check if a position is walkable (not a wall or obstacle)
bool is_walkable(GridWorld* world, int x, int y) {
    if (!world || !is_valid_position(world, x, y)) {
        return false;
    }
    
    CellType cell = world->grid[y][x];
    return (cell != CELL_WALL && cell != CELL_OBSTACLE);
}

// Calculate reward based on the move
float calculate_reward(GridWorld* world, Position old_pos __attribute__((unused)), Position new_pos, bool valid_move) {
    if (!world) return 0.0f;
    
    // If invalid move (hit wall), return wall penalty
    if (!valid_move) {
        return world->wall_penalty;
    }
    
    // If reached goal, return goal reward
    if (positions_equal(new_pos, world->goal_pos)) {
        return world->goal_reward;
    }
    
    // Otherwise, return step penalty (encourages efficiency)
    return world->step_penalty;
}

// Check if two positions are equal
bool positions_equal(Position a, Position b) {
    return (a.x == b.x && a.y == b.y);
}

// Convert 2D position to 1D state index
int position_to_state(GridWorld* world, Position pos) {
    if (!world) return -1;
    return pos.y * world->width + pos.x;
}

// Convert 1D state index to 2D position
Position state_to_position(GridWorld* world, int state) {
    Position pos = {-1, -1};
    if (!world || state < 0 || state >= world->width * world->height) {
        return pos;
    }
    pos.x = state % world->width;
    pos.y = state / world->width;
    return pos;
}

// Get current state structure
State get_current_state(GridWorld* world) {
    State state = {0};
    if (!world) {
        state.state_index = -1;
        state.position = (Position){-1, -1};
        state.is_terminal = true;
        state.is_valid = false;
        return state;
    }
    
    state.state_index = get_state_index(world);
    state.position = world->agent_pos;
    state.is_terminal = is_terminal_state(world, world->agent_pos);
    state.is_valid = is_valid_position(world, world->agent_pos.x, world->agent_pos.y);
    
    return state;
}

// Set cell type at specified position
void set_cell(GridWorld* world, int x, int y, CellType type) {
    if (!world || !is_valid_position(world, x, y)) {
        return;
    }
    world->grid[y][x] = type;
}

// Get cell type at specified position
CellType get_cell(GridWorld* world, int x, int y) {
    if (!world || !is_valid_position(world, x, y)) {
        return CELL_WALL; // Safe default for out-of-bounds
    }
    return world->grid[y][x];
}

// Print environment information
void print_environment_info(GridWorld* world) {
    if (!world) {
        printf("Error: Cannot print info for NULL GridWorld\n");
        return;
    }
    
    printf("\nEnvironment Information:\n");
    printf("========================\n");
    printf("Grid size: %dx%d (%d total states)\n", world->width, world->height, 
           world->width * world->height);
    printf("Agent position: (%d, %d)\n", world->agent_pos.x, world->agent_pos.y);
    printf("Start position: (%d, %d)\n", world->start_pos.x, world->start_pos.y);
    printf("Goal position: (%d, %d)\n", world->goal_pos.x, world->goal_pos.y);
    printf("Rewards: Goal=%.1f, Wall=%.1f, Step=%.1f\n", 
           world->goal_reward, world->wall_penalty, world->step_penalty);
    printf("Max steps per episode: %d\n", world->max_steps);
    printf("Episode status: %s (steps taken: %d)\n", 
           world->episode_done ? "Done" : "Active", world->episode_steps);
    printf("Total reward this episode: %.2f\n", world->total_reward);
    
    // Count different cell types
    int wall_count = 0, obstacle_count = 0;
    for (int y = 0; y < world->height; y++) {
        for (int x = 0; x < world->width; x++) {
            CellType cell = world->grid[y][x];
            if (cell == CELL_WALL) wall_count++;
            else if (cell == CELL_OBSTACLE) obstacle_count++;
        }
    }
    printf("Obstacles: %d walls, %d obstacles\n", wall_count, obstacle_count);
}

// Validate environment configuration
bool validate_environment(GridWorld* world) {
    if (!world) {
        printf("Error: NULL GridWorld\n");
        return false;
    }
    
    // Check basic parameters
    if (world->width <= 0 || world->height <= 0) {
        printf("Error: Invalid grid dimensions: %dx%d\n", world->width, world->height);
        return false;
    }
    
    if (world->max_steps <= 0) {
        printf("Error: Invalid max_steps: %d\n", world->max_steps);
        return false;
    }
    
    // Check positions are within bounds
    if (!is_valid_position(world, world->start_pos.x, world->start_pos.y)) {
        printf("Error: Start position (%d, %d) is out of bounds\n", 
               world->start_pos.x, world->start_pos.y);
        return false;
    }
    
    if (!is_valid_position(world, world->goal_pos.x, world->goal_pos.y)) {
        printf("Error: Goal position (%d, %d) is out of bounds\n", 
               world->goal_pos.x, world->goal_pos.y);
        return false;
    }
    
    if (!is_valid_position(world, world->agent_pos.x, world->agent_pos.y)) {
        printf("Error: Agent position (%d, %d) is out of bounds\n", 
               world->agent_pos.x, world->agent_pos.y);
        return false;
    }
    
    // Check that start and goal positions are walkable
    if (!is_walkable(world, world->start_pos.x, world->start_pos.y)) {
        printf("Error: Start position (%d, %d) is not walkable\n", 
               world->start_pos.x, world->start_pos.y);
        return false;
    }
    
    if (!is_walkable(world, world->goal_pos.x, world->goal_pos.y)) {
        printf("Error: Goal position (%d, %d) is not walkable\n", 
               world->goal_pos.x, world->goal_pos.y);
        return false;
    }
    
    // Check that start and goal are different
    if (positions_equal(world->start_pos, world->goal_pos)) {
        printf("Warning: Start and goal positions are the same\n");
    }
    
    return true;
}

// Validate reward values
bool validate_reward_values(GridWorld* world) {
    if (!world) return false;
    
    // Check that goal reward is positive and larger than penalties
    if (world->goal_reward <= 0) {
        printf("Warning: Goal reward %.2f is not positive\n", world->goal_reward);
        return false;
    }
    
    if (world->goal_reward <= -world->step_penalty) {
        printf("Warning: Goal reward %.2f is not much larger than step penalty %.2f\n", 
               world->goal_reward, world->step_penalty);
    }
    
    if (world->goal_reward <= -world->wall_penalty) {
        printf("Warning: Goal reward %.2f is not much larger than wall penalty %.2f\n", 
               world->goal_reward, world->wall_penalty);
    }
    
    return true;
}
