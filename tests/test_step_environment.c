#include "include/environment.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

// Helper function to compare floats with tolerance
bool float_equals(float a, float b, float tolerance) {
    return fabs(a - b) < tolerance;
}

// Test the step_environment function specifically
int main() {
    printf("=== Testing step_environment Function ===\n");
    
    // Test 1: Basic step_environment functionality
    printf("\nTest 1: Testing basic step_environment functionality...\n");
    GridWorld* world = create_grid_world(5, 5);
    assert(world != NULL);
    
    reset_environment(world);
    
    // Test valid move
    StepResult result = step_environment(world, ACTION_RIGHT);
    assert(result.valid_action == true);
    assert(result.done == false);
    assert(float_equals(result.reward, -1.0f, 0.01f));
    assert(result.next_state.state_index == 1);
    assert(result.next_state.position.x == 1);
    assert(result.next_state.position.y == 0);
    assert(result.next_state.is_valid == true);
    assert(result.next_state.is_terminal == false);
    printf("✓ Valid move working correctly\n");
    
    // Test 2: Invalid move (wall collision)
    printf("\nTest 2: Testing wall collision...\n");
    reset_environment(world);
    
    result = step_environment(world, ACTION_UP);  // Try to move up from (0,0)
    assert(result.valid_action == false);
    assert(result.done == false);
    assert(float_equals(result.reward, -10.0f, 0.01f));
    assert(result.next_state.state_index == 0);  // Should remain at same position
    assert(result.next_state.position.x == 0);
    assert(result.next_state.position.y == 0);
    printf("✓ Wall collision working correctly\n");
    
    // Test 3: Reaching goal
    printf("\nTest 3: Testing goal achievement...\n");
    world->agent_pos.x = 3;
    world->agent_pos.y = 4;
    
    result = step_environment(world, ACTION_RIGHT);  // Move to goal at (4,4)
    assert(result.valid_action == true);
    assert(result.done == true);  // Episode should be done
    assert(float_equals(result.reward, 100.0f, 0.01f));
    assert(result.next_state.state_index == 24);  // 4*5 + 4 = 24
    assert(result.next_state.position.x == 4);
    assert(result.next_state.position.y == 4);
    assert(result.next_state.is_terminal == true);
    printf("✓ Goal achievement working correctly\n");
    
    // Test 4: Error handling
    printf("\nTest 4: Testing error handling...\n");
    
    // Test NULL world
    result = step_environment(NULL, ACTION_UP);
    assert(result.valid_action == false);
    assert(result.done == true);
    assert(result.next_state.state_index == -1);
    assert(result.next_state.is_valid == false);
    
    // Test invalid action
    reset_environment(world);
    result = step_environment(world, (Action)999);  // Invalid action
    assert(result.valid_action == false);
    assert(result.done == false);  // Episode not done due to invalid action
    assert(float_equals(result.reward, 0.0f, 0.01f));
    
    // Test step on completed episode
    world->episode_done = true;
    result = step_environment(world, ACTION_RIGHT);
    assert(result.valid_action == false);
    assert(result.done == true);
    assert(float_equals(result.reward, 0.0f, 0.01f));
    
    printf("✓ Error handling working correctly\n");
    
    // Test 5: Compare with original step function
    printf("\nTest 5: Comparing step_environment with step function...\n");
    reset_environment(world);
    
    float step_reward;
    int step_state = step(world, ACTION_RIGHT, &step_reward);
    
    reset_environment(world);
    StepResult step_env_result = step_environment(world, ACTION_RIGHT);
    
    // Results should be equivalent
    assert(step_state == step_env_result.next_state.state_index);
    assert(float_equals(step_reward, step_env_result.reward, 0.01f));
    
    printf("✓ Consistency with step function verified\n");
    
    // Test 6: State conversion functions
    printf("\nTest 6: Testing state conversion functions...\n");
    
    Position pos = {2, 3};
    int state_idx = position_to_state(world, pos);
    assert(state_idx == 17);  // 3*5 + 2 = 17
    
    Position converted_pos = state_to_position(world, state_idx);
    assert(converted_pos.x == pos.x);
    assert(converted_pos.y == pos.y);
    
    // Test invalid conversions
    int invalid_state = position_to_state(world, (Position){-1, 0});
    assert(invalid_state == -1);
    
    Position invalid_pos = state_to_position(world, -1);
    assert(invalid_pos.x == -1 && invalid_pos.y == -1);
    
    printf("✓ State conversion functions working correctly\n");
    
    // Cleanup
    destroy_grid_world(world);
    
    printf("\n🎉 All step_environment tests passed!\n");
    printf("Tested features:\n");
    printf("  ✅ Basic step_environment functionality\n");
    printf("  ✅ Wall collision handling\n");
    printf("  ✅ Goal achievement detection\n");
    printf("  ✅ Error handling and edge cases\n");
    printf("  ✅ Consistency with original step function\n");
    printf("  ✅ State conversion utilities\n");
    
    return 0;
}