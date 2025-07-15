#include "include/environment.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

// Helper function to compare floats with tolerance
bool float_equals(float a, float b, float tolerance) {
    return fabs(a - b) < tolerance;
}

// Test comprehensive reward calculation scenarios
int main() {
    printf("=== Testing Comprehensive Reward Structure ===\n");
    
    // Test 1: Default reward configuration
    printf("\nTest 1: Testing default reward configuration...\n");
    GridWorld* world = create_grid_world(5, 5);
    assert(world != NULL);
    assert(float_equals(world->goal_reward, 100.0f, 0.01f));
    assert(float_equals(world->wall_penalty, -10.0f, 0.01f));
    assert(float_equals(world->step_penalty, -1.0f, 0.01f));
    printf("✓ Default rewards: goal=%.1f, wall=%.1f, step=%.1f\n",
           world->goal_reward, world->wall_penalty, world->step_penalty);
    
    // Test 2: Reward calculation for goal reached
    printf("\nTest 2: Testing goal reward calculation...\n");
    reset_environment(world);
    // Move agent to position adjacent to goal
    world->agent_pos.x = 3;
    world->agent_pos.y = 4;
    
    Position old_pos = world->agent_pos;
    Position goal_pos = {4, 4};
    float reward = calculate_reward(world, old_pos, goal_pos, true);
    assert(float_equals(reward, 100.0f, 0.01f));
    printf("✓ Goal reward calculation: %.1f\n", reward);
    
    // Test 3: Reward calculation for wall collision
    printf("\nTest 3: Testing wall penalty calculation...\n");
    reset_environment(world);
    Position boundary_pos = {0, 0};
    Position invalid_pos = {-1, 0}; // Outside boundary
    reward = calculate_reward(world, boundary_pos, boundary_pos, false);
    assert(float_equals(reward, -10.0f, 0.01f));
    printf("✓ Wall penalty calculation: %.1f\n", reward);
    
    // Test 4: Reward calculation for empty space movement
    printf("\nTest 4: Testing step penalty calculation...\n");
    Position empty_pos1 = {1, 1};
    Position empty_pos2 = {2, 1};
    reward = calculate_reward(world, empty_pos1, empty_pos2, true);
    assert(float_equals(reward, -1.0f, 0.01f));
    printf("✓ Step penalty calculation: %.1f\n", reward);
    
    // Test 5: Configuration-based grid world creation
    printf("\nTest 5: Testing create_grid_world_from_config...\n");
    EnvironmentConfig config = {
        .width = 8,
        .height = 6,
        .step_penalty = -0.5f,
        .goal_reward = 150.0f,
        .wall_penalty = -15.0f,
        .max_steps = 100,
        .stochastic = false,
        .action_noise = 0.0f
    };
    
    GridWorld* config_world = create_grid_world_from_config(config);
    assert(config_world != NULL);
    assert(config_world->width == 8);
    assert(config_world->height == 6);
    assert(float_equals(config_world->goal_reward, 150.0f, 0.01f));
    assert(float_equals(config_world->wall_penalty, -15.0f, 0.01f));
    assert(float_equals(config_world->step_penalty, -0.5f, 0.01f));
    assert(config_world->max_steps == 100);
    printf("✓ Configuration-based creation successful\n");
    
    // Test 6: Reward validation
    printf("\nTest 6: Testing reward validation...\n");
    assert(validate_reward_values(world) == true);
    assert(validate_reward_values(config_world) == true);
    printf("✓ Valid reward configurations pass validation\n");
    
    // Test 7: Invalid reward configurations
    printf("\nTest 7: Testing invalid reward configurations...\n");
    GridWorld* invalid_world = create_grid_world(3, 3);
    invalid_world->goal_reward = -50.0f;  // Invalid: goal should be positive
    assert(validate_reward_values(invalid_world) == false);
    
    invalid_world->goal_reward = 100.0f;
    invalid_world->wall_penalty = 5.0f;   // Invalid: wall penalty should be negative
    assert(validate_reward_values(invalid_world) == false);
    
    invalid_world->wall_penalty = -10.0f;
    invalid_world->step_penalty = 2.0f;   // Invalid: step penalty should be negative
    assert(validate_reward_values(invalid_world) == false);
    printf("✓ Invalid reward configurations properly rejected\n");
    
    // Test 8: Dynamic reward value setting
    printf("\nTest 8: Testing dynamic reward value setting...\n");
    bool success = set_reward_values(world, 200.0f, -20.0f, -2.0f);
    assert(success == true);
    assert(float_equals(world->goal_reward, 200.0f, 0.01f));
    assert(float_equals(world->wall_penalty, -20.0f, 0.01f));
    assert(float_equals(world->step_penalty, -2.0f, 0.01f));
    
    // Test invalid setting (should fail and rollback)
    success = set_reward_values(world, -50.0f, 10.0f, 5.0f); // All invalid
    assert(success == false);
    // Values should remain unchanged
    assert(float_equals(world->goal_reward, 200.0f, 0.01f));
    assert(float_equals(world->wall_penalty, -20.0f, 0.01f));
    assert(float_equals(world->step_penalty, -2.0f, 0.01f));
    printf("✓ Dynamic reward setting with validation working correctly\n");
    
    // Test 9: Reward value retrieval
    printf("\nTest 9: Testing reward value retrieval...\n");
    float goal, wall, step_val;
    get_reward_values(world, &goal, &wall, &step_val);
    assert(float_equals(goal, 200.0f, 0.01f));
    assert(float_equals(wall, -20.0f, 0.01f));
    assert(float_equals(step_val, -2.0f, 0.01f));
    printf("✓ Reward value retrieval working correctly\n");
    
    // Test 10: End-to-end reward scenario testing
    printf("\nTest 10: Testing end-to-end reward scenarios...\n");
    reset_environment(world);
    float total_reward = 0.0f;
    
    // Scenario: Agent moves right 2 steps, hits wall, then reaches goal
    float reward_step;
    
    // Step 1: Move right (empty space)
    step(world, ACTION_RIGHT, &reward_step);
    total_reward += reward_step;
    assert(float_equals(reward_step, -2.0f, 0.01f)); // step penalty
    
    // Step 2: Move right again (empty space)
    step(world, ACTION_RIGHT, &reward_step);
    total_reward += reward_step;
    assert(float_equals(reward_step, -2.0f, 0.01f)); // step penalty
    
    // Step 3: Try to move up (hit boundary)
    step(world, ACTION_UP, &reward_step);
    total_reward += reward_step;
    assert(float_equals(reward_step, -20.0f, 0.01f)); // wall penalty
    
    // Move agent to near goal for final test
    world->agent_pos.x = 3;
    world->agent_pos.y = 4;
    
    // Step 4: Move to goal
    step(world, ACTION_RIGHT, &reward_step);
    total_reward += reward_step;
    assert(float_equals(reward_step, 200.0f, 0.01f)); // goal reward
    
    float expected_total = -2.0f + -2.0f + -20.0f + 200.0f; // 176.0f
    assert(float_equals(total_reward, expected_total, 0.01f));
    printf("✓ End-to-end scenario: total reward = %.1f (expected %.1f)\n", 
           total_reward, expected_total);
    
    // Test 11: Error handling
    printf("\nTest 11: Testing error handling...\n");
    
    // Test NULL parameters
    assert(validate_reward_values(NULL) == false);
    assert(set_reward_values(NULL, 100.0f, -10.0f, -1.0f) == false);
    
    float dummy;
    get_reward_values(NULL, &dummy, &dummy, &dummy); // Should handle gracefully
    get_reward_values(world, NULL, &dummy, &dummy);  // Should handle gracefully
    
    // Test invalid config
    EnvironmentConfig invalid_config = {
        .width = -1,  // Invalid
        .height = 5,
        .step_penalty = -1.0f,
        .goal_reward = 100.0f,
        .wall_penalty = -10.0f,
        .max_steps = 50
    };
    GridWorld* null_world = create_grid_world_from_config(invalid_config);
    assert(null_world == NULL);
    printf("✓ Error handling working correctly\n");
    
    // Cleanup
    destroy_grid_world(world);
    destroy_grid_world(config_world);
    destroy_grid_world(invalid_world);
    
    printf("\n🎉 All reward system tests passed!\n");
    printf("Tested features:\n");
    printf("  ✅ Default reward configuration (+100, -10, -1)\n");
    printf("  ✅ Goal reward calculation (+100 for reaching goal)\n");
    printf("  ✅ Wall penalty calculation (-10 for hitting walls)\n");
    printf("  ✅ Step penalty calculation (-1 for empty space movement)\n");
    printf("  ✅ Configuration-based world creation\n");
    printf("  ✅ Reward value validation\n");
    printf("  ✅ Dynamic reward configuration\n");
    printf("  ✅ End-to-end reward scenarios\n");
    printf("  ✅ Error handling and edge cases\n");
    
    return 0;
}