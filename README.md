# Reinforcement Learning Agent Environment with Raylib

A comprehensive C implementation of a reinforcement learning environment using Q-Learning algorithms with real-time visualization powered by Raylib.

## Project Overview

This project provides a complete framework for creating and training reinforcement learning agents in grid-world environments. The modular architecture separates core RL logic from visualization, making it easy to extend and modify for different scenarios.

## Core Data Structures

### Agent System (`include/agent.h`)

#### Action Enumeration
```c
typedef enum {
    ACTION_UP = 0,
    ACTION_DOWN = 1,
    ACTION_LEFT = 2,
    ACTION_RIGHT = 3,
    NUM_ACTIONS = 4
} Action;
```

#### Q-Learning Agent
```c
typedef struct {
    float** q_table;        // Q(state, action) values
    int num_states;         // Total number of states
    int num_actions;        // Total number of actions
    float learning_rate;    // Alpha (α)
    float discount_factor;  // Gamma (γ)
    float epsilon;          // Exploration rate
    float epsilon_decay;    // Epsilon decay rate
    float epsilon_min;      // Minimum epsilon value
    int current_state;      // Current state index
    Action last_action;     // Last action taken
} QLearningAgent;
```

#### Experience Replay
```c
typedef struct {
    int state;
    Action action;
    float reward;
    int next_state;
    bool done;
} Experience;

typedef struct {
    Experience* experiences;
    int capacity;
    int size;
    int current_index;
} ExperienceBuffer;
```

### Environment System (`include/environment.h`)

#### Grid World Structure
```c
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
```

#### Comprehensive Reward Structure

The environment implements a carefully designed reward system to encourage optimal pathfinding:

- **Goal Reached**: `+100` points - Large positive reward for task completion
- **Wall Collision**: `-10` points - Moderate penalty for invalid moves  
- **Empty Space Movement**: `-1` point - Small penalty encouraging efficiency

The reward values are fully configurable and include validation to ensure effective learning. See `REWARD_DESIGN.md` for detailed design rationale.

#### Cell Types
```c
typedef enum {
    CELL_EMPTY = 0,
    CELL_WALL = 1,
    CELL_GOAL = 2,
    CELL_AGENT = 3,
    CELL_OBSTACLE = 4,
    CELL_START = 5
} CellType;
```

#### State Representation
```c
typedef struct {
    int state_index;           // 1D state representation
    Position position;         // 2D position representation
    bool is_terminal;          // Whether this is a terminal state
    bool is_valid;             // Whether this state is valid
} State;
```

### Visualization System (`include/rendering.h`)

#### Rendering Configuration
```c
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
```

#### Color Scheme
```c
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
```

### Utility System (`include/utils.h`)

#### Random Number Generation
```c
typedef struct {
    unsigned int seed;
    bool initialized;
} RandomState;
```

#### Performance Monitoring
```c
typedef struct {
    clock_t start_time;
    clock_t end_time;
    double elapsed_seconds;
    bool is_running;
} Timer;
```

#### Configuration Management
```c
typedef struct {
    char key[64];
    char value[256];
} ConfigPair;

typedef struct {
    ConfigPair* pairs;
    int count;
    int capacity;
} ConfigFile;
```

## Project Structure

```
c_raylib_simulation/
├── include/                    # Header files
│   ├── agent.h                # RL agent structures and functions
│   ├── environment.h          # Grid world environment
│   ├── rendering.h            # Raylib visualization
│   └── utils.h                # Utility functions and helpers
├── src/                       # Source files
│   ├── main.c                 # Main application entry point
│   ├── agent.c                # Q-learning implementation
│   ├── environment.c          # Grid world logic
│   ├── rendering.c            # Visualization implementation
│   └── utils.c                # Utility function implementations
├── build/                     # Compiled object files (created during build)
├── bin/                       # Final executable (created during build)
├── Makefile                   # Build configuration
├── README.md                  # This file
└── claude_notes.md           # Development notes and plans
```

## Building the Project

### Prerequisites

1. **C Compiler**: GCC or Clang
2. **Raylib**: Graphics library for visualization
3. **Make**: Build system

### Installation

#### macOS (with Homebrew)
```bash
# Install dependencies
brew install raylib gcc make

# Build project
make all
```

#### Linux (Ubuntu/Debian)
```bash
# Install dependencies
sudo apt-get update
sudo apt-get install build-essential libraylib-dev

# Build project
make all
```

#### Windows (MinGW)
```bash
# Install raylib manually or use vcpkg
# Then build with:
make all
```

### Build Commands

```bash
# Basic build
make all

# Debug build with symbols
make debug

# Optimized release build
make release

# Clean build artifacts
make clean

# Build and run
make run

# Check dependencies
make check-deps

# Show all available commands
make help
```

## Key Features

### Reinforcement Learning
- **Q-Learning Algorithm**: Classic temporal difference learning
- **Epsilon-Greedy Policy**: Balanced exploration and exploitation
- **Experience Replay**: Optional experience buffer for improved learning
- **Configurable Parameters**: Learning rate, discount factor, exploration rate

### Environment
- **Grid World**: Customizable 2D grid environments
- **Multiple Cell Types**: Empty, walls, goals, obstacles, start positions
- **Reward System**: Configurable rewards and penalties
- **State Management**: Automatic state indexing and conversion
- **Maze Generation**: Built-in maze generation algorithms

### Visualization
- **Real-time Rendering**: Live visualization of agent training
- **Q-value Display**: Heat maps and arrow visualization of Q-values
- **Agent Trail**: Visual trail of agent movement
- **Statistics Panel**: Real-time training metrics
- **Interactive Controls**: Pause, reset, speed control
- **Multiple Color Schemes**: Customizable visual themes

### Utilities
- **Memory Management**: Safe allocation and tracking
- **Configuration Files**: Load/save settings
- **Performance Monitoring**: Timing and profiling tools
- **Logging System**: Comprehensive logging with levels
- **Data Export**: CSV and JSON export for analysis

## Usage Example

```c
#include "agent.h"
#include "environment.h"
#include "rendering.h"

int main() {
    // Create environment
    GridWorld* world = create_grid_world(10, 10);
    
    // Create agent
    QLearningAgent* agent = create_agent(
        world->width * world->height,  // num_states
        NUM_ACTIONS,                   // num_actions
        0.1f,                         // learning_rate
        0.95f,                        // discount_factor
        1.0f                          // epsilon
    );
    
    // Initialize visualization
    VisualizationState* vis = init_visualization(800, 600, 40);
    
    // Training loop
    for (int episode = 0; episode < 1000; episode++) {
        reset_environment(world);
        
        while (!world->episode_done) {
            int state = position_to_state(world, world->agent_pos);
            Action action = select_action(agent, state);
            StepResult result = step_environment(world, action);
            
            update_q_value(agent, state, action, result.reward, 
                          result.next_state.state_index, result.done);
            
            // Render frame
            render_frame(vis, world, agent, NULL);
        }
        
        decay_epsilon(agent);
    }
    
    // Cleanup
    destroy_agent(agent);
    destroy_grid_world(world);
    destroy_visualization(vis);
    
    return 0;
}
```

## Configuration

The system supports configuration files for easy parameter tuning:

```ini
# config.ini
[environment]
width=15
height=15
goal_reward=100.0
step_penalty=-1.0
wall_penalty=-10.0
max_steps=300

[agent]
learning_rate=0.1
discount_factor=0.95
epsilon=1.0
epsilon_decay=0.995
epsilon_min=0.01

[rendering]
cell_size=30
show_q_values=true
show_trail=true
animation_speed=0.5
```

### Reward Configuration

The reward system supports both static and dynamic configuration:

```c
// Static configuration at creation
EnvironmentConfig config = {
    .width = 15, .height = 15,
    .goal_reward = 150.0f,
    .wall_penalty = -15.0f,
    .step_penalty = -0.5f,
    .max_steps = 300
};
GridWorld* world = create_grid_world_from_config(config);

// Dynamic configuration during runtime
set_reward_values(world, 200.0f, -20.0f, -2.0f);
```

## Performance Considerations

- **Memory Usage**: Q-table size scales as O(states × actions)
- **Rendering**: Can be disabled for faster training
- **Experience Replay**: Optional for memory-efficient training
- **State Representation**: Efficient 1D indexing of 2D positions

## Extension Points

The modular design allows easy extension:

1. **New Algorithms**: Implement SARSA, Deep Q-Learning, etc.
2. **Different Environments**: Continuous spaces, multi-agent, etc.
3. **Enhanced Visualization**: 3D rendering, web interface, etc.
4. **Advanced Features**: Neural networks, policy gradients, etc.

## Contributing

1. Follow the existing code structure and naming conventions
2. Add appropriate documentation for new features
3. Update this README when adding new data structures
4. Test thoroughly before submitting changes

## License

This project is provided as-is for educational and research purposes.

## Next Steps

Based on the comprehensive data structures defined, the next development phases would be:

1. **Implement Core Functions**: Fill in the function implementations
2. **Basic Training Loop**: Create a simple training scenario
3. **Visualization Integration**: Connect Raylib rendering
4. **Testing and Debugging**: Validate the Q-learning implementation
5. **Advanced Features**: Add experience replay, neural networks, etc.

The foundation is now in place for a robust and extensible RL environment!
