I'll create a comprehensive step-by-step plan to build a reinforcement learning agent in C using Raylib for a 2D grid-world environment.

## Step-by-Step Implementation Plan

### Phase 1: Set Up the Development Environment (Completed)

1. **Install Dependencies**
   - Install Raylib library for your platform
   - Set up a C compiler (gcc/clang)
   - Create project directory structure:
   ```
   rl_agent/
   ├── src/
   │   ├── main.c
   │   ├── environment.c
   │   ├── agent.c
   │   ├── rendering.c
   │   └── utils.c
   ├── include/
   │   ├── environment.h
   │   ├── agent.h
   │   ├── rendering.h
   │   └── utils.h
   └── Makefile
   ```

### Phase 2: Define Core Data Structures (Completed)

2. **Create Basic Types and Structures**
   ```c
   // environment.h
   typedef struct {
       int x, y;
   } Position;
   
   typedef struct {
       int width, height;
       int** grid;  // 0=empty, 1=wall, 2=goal, 3=agent
       Position agent_pos;
       Position goal_pos;
   } GridWorld;
   
   // agent.h
   typedef enum {
       ACTION_UP = 0,
       ACTION_DOWN,
       ACTION_LEFT,
       ACTION_RIGHT,
       NUM_ACTIONS
   } Action;
   
   typedef struct {
       float** q_table;  // Q(state, action)
       float learning_rate;
       float discount_factor;
       float epsilon;  // for epsilon-greedy
   } QLearningAgent;
   ```

### Phase 3: Implement the Environment (Completed)

3. **Create Environment Functions** (Completed)
   - `GridWorld* create_grid_world(int width, int height)` (Completed)
   - `void reset_environment(GridWorld* world)` 
   - `int step(GridWorld* world, Action action, float* reward)`
   - `int get_state_index(GridWorld* world)` // Convert 2D position to 1D state
   - `bool is_terminal_state(GridWorld* world)`
   - `void destroy_grid_world(GridWorld* world)`

4. **Define Reward Structure** (Completed)
   - Reaching goal: +100
   - Hitting wall: -10
   - Empty space: -1 (encourage finding shortest path)

### Phase 4: Implement the RL Agent (Completed)

5. **Create Q-Learning Agent**
   - `QLearningAgent* create_agent(int num_states, int num_actions)`
   - `Action select_action(QLearningAgent* agent, int state)` // Epsilon-greedy
   - `void update_q_value(QLearningAgent* agent, int state, Action action, float reward, int next_state)`
   - `Action get_best_action(QLearningAgent* agent, int state)`
   - `void destroy_agent(QLearningAgent* agent)`

6. **Implement Q-Learning Update Rule**
   ```c
   Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
   ```

### Phase 5: Create Visualization with Raylib (Completed)

7. **Set Up Rendering Functions** (Completed)
   - `void init_graphics(int screen_width, int screen_height)`
   - `void draw_grid_world(GridWorld* world, int cell_size)`
   - `void draw_agent(Position pos, int cell_size)`
   - `void draw_goal(Position pos, int cell_size)`
   - `void draw_walls(GridWorld* world, int cell_size)`
   - `void draw_q_values(QLearningAgent* agent, GridWorld* world, int cell_size)`

8. **Add Visual Debugging Features** (not Completed)
   - Show Q-values as heat map or arrows
   - Display current episode and total reward
   - Show agent's chosen action
   - Visualize exploration vs exploitation

### Phase 6: Implement Training Loop (Completed)

9. **Create Main Training Structure**
   ```c
   // main.c structure
   - Initialize environment and agent
   - For each episode:
     - Reset environment
     - While not terminal:
       - Select action (epsilon-greedy)
       - Take action, get reward
       - Update Q-values
       - Render if in visualization mode
     - Decay epsilon
   - Save learned policy
   ```

### Phase 7: Add Control and Monitoring

10. **Implement User Controls** (Completed)
    - Space: Pause/Resume training
    - R: Reset and restart
    - V: Toggle Q-value visualization
    - +/-: Adjust training speed
    - S: Save current Q-table
    - L: Load saved Q-table

11. **Add Performance Metrics** (Completed)
    - Track average reward per episode
    - Count steps to reach goal
    - Display learning curves
    - Show convergence indicators

### Phase 8: Optimize and Extend

12. **Performance Improvements**
    - Implement experience replay buffer (Completed)
    - Add priority to state visits (completed)
    - Optimize Q-table access patterns (Completed)
    - Add multi-threading for batch updates

13. **Environment Extensions**
    - Add multiple goals
    - Implement moving obstacles
    - Create maze generation algorithms
    - Add different reward schemes

### Phase 9: Testing and Debugging

14. **Create Test Scenarios**
    - Simple path finding (no obstacles)
    - Maze navigation
    - Dynamic obstacles
    - Multiple goal scenarios

15. **Debugging Tools**
    - Log state transitions
    - Visualize decision making process
    - Track Q-value convergence
    - Validate action selection

### Implementation Tips:

**Memory Management**
- Use dynamic allocation for Q-table based on grid size
- Free all allocated memory properly
- Consider using memory pools for frequent allocations

**Separation of Concerns**
- Keep RL logic completely separate from rendering
- Use callbacks or flags to control visualization
- Make the agent module reusable for different environments

**Visualization Best Practices**
- Use different colors for agent, goal, walls, and empty spaces
- Animate agent movement smoothly
- Show trails of recent paths
- Use transparency for Q-value overlays

This modular approach allows you to start with a basic implementation and gradually add complexity while maintaining clean, debuggable code. The visualization with Raylib will help you understand how your agent learns and make debugging much easier.