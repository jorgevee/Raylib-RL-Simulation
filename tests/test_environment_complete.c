#include "include/environment.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

// Comprehensive test program for all GridWorld environment functions
int main() {
    printf("=== Testing GridWorld Environment Functions ===\n");
    
    // Test 1: create_grid_world
    printf("\nTest 1: Testing create_grid_world...\n");
    GridWorld* world = create_grid_world(5, 5);
    assert(world != NULL);
    assert(world->width == 5);
    assert(world->height == 5);
    assert(world->agent_pos.x == 0);
    assert(world->agent_pos.y == 0);
    assert(world->goal_pos.x == 4);
    assert(world->goal_pos.y == 4);
    printf("✓ create_grid_world working correctly\n");
    
    // Test 2: get_state_index
    printf("\nTest 2: Testing get_state_index...\n");
    int state = get_state_index(world);
    assert(state == 0); // (0,0) in 5x5 grid = 0*5 + 0 = 0
    world->agent_pos.x = 2;
    world->agent_pos.y = 3;
    state = get_state_index(world);
    assert(state == 17); // (2,3) in 5x5 grid = 3*5 + 2 = 17
    printf("✓ get_state_index working correctly\n");
    
    // Test 3: is_terminal_state
    printf("\nTest 3: Testing is_terminal_state...\n");
    Position test_pos = {2, 3};
    bool terminal = is_terminal_state(world, test_pos);
    assert(terminal == false); // Not at goal
    
    Position goal_pos = {4, 4};
    terminal = is_terminal_state(world, goal_pos);
    assert(terminal == true); // At goal
    printf("✓ is_terminal_state working correctly\n");
    
    // Test 4: reset_environment
    printf("\nTest 4: Testing reset_environment...\n");
    world->agent_pos.x = 3;
    world->agent_pos.y = 2;
    world->episode_steps = 10;
    world->episode_done = true;
    world->total_reward = 50.0f;
    
    reset_environment(world);
    
    assert(world->agent_pos.x == 0);
    assert(world->agent_pos.y == 0);
    assert(world->episode_steps == 0);
    assert(world->episode_done == false);
    assert(world->total_reward == 0.0f);
    printf("✓ reset_environment working correctly\n");
    
    // Test 5: step function
    printf("\nTest 5: Testing step function...\n");
    float reward;
    
    // Test moving right from (0,0) to (1,0)
    int new_state = step(world, ACTION_RIGHT, &reward);
    assert(world->agent_pos.x == 1);
    assert(world->agent_pos.y == 0);
    assert(reward == -1.0f); // step penalty
    assert(new_state == 1); // new state index
    assert(world->episode_steps == 1);
    assert(world->episode_done == false);
    printf("✓ Valid move working correctly\n");
    
    // Test moving up from (1,0) - should hit boundary and stay at (1,0)
    new_state = step(world, ACTION_UP, &reward);
    assert(world->agent_pos.x == 1);
    assert(world->agent_pos.y == 0); // Should stay same due to boundary
    assert(reward == -10.0f); // wall penalty
    assert(new_state == 1); // same state
    assert(world->episode_steps == 2);
    printf("✓ Boundary collision working correctly\n");
    
    // Test reaching goal
    world->agent_pos.x = 3;
    world->agent_pos.y = 4;
    new_state = step(world, ACTION_RIGHT, &reward);
    assert(world->agent_pos.x == 4);
    assert(world->agent_pos.y == 4);
    assert(reward == 100.0f); // goal reward
    assert(world->episode_done == true);
    printf("✓ Goal reaching working correctly\n");
    
    // Test 6: Helper functions
    printf("\nTest 6: Testing helper functions...\n");
    
    // Test is_valid_position
    assert(is_valid_position(world, 0, 0) == true);
    assert(is_valid_position(world, 4, 4) == true);
    assert(is_valid_position(world, -1, 0) == false);
    assert(is_valid_position(world, 0, -1) == false);
    assert(is_valid_position(world, 5, 0) == false);
    assert(is_valid_position(world, 0, 5) == false);
    printf("✓ is_valid_position working correctly\n");
    
    // Test is_walkable
    assert(is_walkable(world, 1, 1) == true); // empty cell
    assert(is_walkable(world, 0, 0) == true); // start cell (walkable)
    assert(is_walkable(world, 4, 4) == true); // goal cell (walkable)
    printf("✓ is_walkable working correctly\n");
    
    // Test positions_equal
    Position pos1 = {2, 3};
    Position pos2 = {2, 3};
    Position pos3 = {2, 4};
    assert(positions_equal(pos1, pos2) == true);
    assert(positions_equal(pos1, pos3) == false);
    printf("✓ positions_equal working correctly\n");
    
    // Test 7: destroy_grid_world
    printf("\nTest 7: Testing destroy_grid_world...\n");
    destroy_grid_world(world);
    printf("✓ destroy_grid_world working correctly\n");
    
    // Test 8: Error handling
    printf("\nTest 8: Testing error handling...\n");
    
    // Test with NULL world
    int error_state = get_state_index(NULL);
    assert(error_state == -1);
    
    bool error_terminal = is_terminal_state(NULL, pos1);
    assert(error_terminal == true);
    
    int error_step = step(NULL, ACTION_UP, &reward);
    assert(error_step == -1);
    
    printf("✓ Error handling working correctly\n");
    
    printf("\n🎉 All tests passed! All GridWorld functions are working correctly.\n");
    printf("Functions tested:\n");
    printf("  - create_grid_world\n");
    printf("  - reset_environment\n");
    printf("  - step\n");
    printf("  - get_state_index\n");
    printf("  - is_terminal_state\n");
    printf("  - destroy_grid_world\n");
    printf("  - Helper functions (is_valid_position, is_walkable, positions_equal, etc.)\n");
    
    return 0;
}
