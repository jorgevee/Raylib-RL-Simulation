#include "include/environment.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

// Simple test program for the create_grid_world function
int main() {
    printf("Testing create_grid_world function...\n");
    
    // Test 1: Normal grid creation
    printf("\nTest 1: Creating a 5x5 grid...\n");
    GridWorld* world = create_grid_world(5, 5);
    assert(world != NULL);
    assert(world->width == 5);
    assert(world->height == 5);
    assert(world->agent_pos.x == 0);
    assert(world->agent_pos.y == 0);
    assert(world->goal_pos.x == 4);
    assert(world->goal_pos.y == 4);
    assert(world->episode_steps == 0);
    assert(world->episode_done == false);
    assert(world->total_reward == 0.0f);
    assert(world->max_steps == 50); // 5*5*2
    printf("✓ 5x5 grid created successfully with correct parameters\n");
    
    // Test 2: Check grid initialization
    printf("\nTest 2: Checking grid cell initialization...\n");
    assert(world->grid[0][0] == CELL_START);
    assert(world->grid[4][4] == CELL_GOAL);
    // Check some empty cells
    assert(world->grid[1][1] == CELL_EMPTY);
    assert(world->grid[2][3] == CELL_EMPTY);
    printf("✓ Grid cells initialized correctly\n");
    
    // Test 3: Different grid size
    printf("\nTest 3: Creating a 10x8 grid...\n");
    GridWorld* world2 = create_grid_world(10, 8);
    assert(world2 != NULL);
    assert(world2->width == 10);
    assert(world2->height == 8);
    assert(world2->goal_pos.x == 9);
    assert(world2->goal_pos.y == 7);
    assert(world2->max_steps == 160); // 10*8*2
    printf("✓ 10x8 grid created successfully\n");
    
    // Test 4: Error handling - invalid dimensions
    printf("\nTest 4: Testing error handling with invalid dimensions...\n");
    GridWorld* invalid1 = create_grid_world(0, 5);
    assert(invalid1 == NULL);
    GridWorld* invalid2 = create_grid_world(5, -1);
    assert(invalid2 == NULL);
    printf("✓ Invalid dimensions properly rejected\n");
    
    // Test 5: Check reward configuration
    printf("\nTest 5: Checking default reward configuration...\n");
    assert(world->step_penalty == -1.0f);
    assert(world->goal_reward == 100.0f);
    assert(world->wall_penalty == -10.0f);
    printf("✓ Default rewards configured correctly\n");
    
    // Cleanup
    // Note: We'll implement destroy_grid_world later, for now just free main structure
    if (world) {
        for (int y = 0; y < world->height; y++) {
            free(world->grid[y]);
        }
        free(world->grid);
        free(world);
    }
    if (world2) {
        for (int y = 0; y < world2->height; y++) {
            free(world2->grid[y]);
        }
        free(world2->grid);
        free(world2);
    }
    
    printf("\n🎉 All tests passed! create_grid_world function is working correctly.\n");
    return 0;
}
