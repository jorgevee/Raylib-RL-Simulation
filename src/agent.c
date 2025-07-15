#include "agent.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>

// Create a new Q-learning agent
QLearningAgent* create_agent(int num_states, int num_actions, float learning_rate, float discount_factor, float epsilon) {
    QLearningAgent* agent = (QLearningAgent*)malloc(sizeof(QLearningAgent));
    if (!agent) {
        fprintf(stderr, "Error: Failed to allocate memory for agent\n");
        return NULL;
    }

    agent->num_states = num_states;
    agent->num_actions = num_actions;
    agent->learning_rate = learning_rate;
    agent->discount_factor = discount_factor;
    agent->epsilon = epsilon;
    agent->epsilon_decay = 0.995f;  // Default decay rate
    agent->epsilon_min = 0.01f;     // Minimum exploration rate
    agent->current_state = 0;
    agent->last_action = ACTION_UP;

    // Allocate Q-table
    agent->q_table = (float**)malloc(num_states * sizeof(float*));
    if (!agent->q_table) {
        fprintf(stderr, "Error: Failed to allocate memory for Q-table\n");
        free(agent);
        return NULL;
    }

    for (int i = 0; i < num_states; i++) {
        agent->q_table[i] = (float*)calloc(num_actions, sizeof(float));
        if (!agent->q_table[i]) {
            fprintf(stderr, "Error: Failed to allocate memory for Q-table row %d\n", i);
            // Clean up previously allocated rows
            for (int j = 0; j < i; j++) {
                free(agent->q_table[j]);
            }
            free(agent->q_table);
            free(agent);
            return NULL;
        }
    }

    return agent;
}

// Destroy the agent and free memory
void destroy_agent(QLearningAgent* agent) {
    if (!agent) return;

    if (agent->q_table) {
        for (int i = 0; i < agent->num_states; i++) {
            free(agent->q_table[i]);
        }
        free(agent->q_table);
    }
    free(agent);
}

// Select action using epsilon-greedy strategy
Action select_action(QLearningAgent* agent, int state) {
    if (!agent || state < 0 || state >= agent->num_states) {
        return ACTION_UP; // Default action
    }

    agent->current_state = state;

    // Epsilon-greedy action selection
    float random_value = (float)rand() / RAND_MAX;
    
    if (random_value < agent->epsilon) {
        // Explore: choose random action
        return (Action)(rand() % agent->num_actions);
    } else {
        // Exploit: choose greedy action
        return select_greedy_action(agent, state);
    }
}

// Select the best action (greedy) for a given state
Action select_greedy_action(QLearningAgent* agent, int state) {
    if (!agent || state < 0 || state >= agent->num_states) {
        return ACTION_UP; // Default action
    }

    Action best_action = ACTION_UP;
    float best_q_value = agent->q_table[state][0];

    // Find action with highest Q-value
    for (int action = 1; action < agent->num_actions; action++) {
        if (agent->q_table[state][action] > best_q_value) {
            best_q_value = agent->q_table[state][action];
            best_action = (Action)action;
        }
    }

    return best_action;
}

// Update Q-value using Q-learning formula
void update_q_value(QLearningAgent* agent, int state, Action action, float reward, int next_state, bool done) {
    if (!agent || state < 0 || state >= agent->num_states || 
        next_state < 0 || next_state >= agent->num_states ||
        (int)action < 0 || (int)action >= agent->num_actions) {
        return;
    }

    float current_q = agent->q_table[state][action];
    float max_next_q = 0.0f;

    // If not terminal state, find maximum Q-value for next state
    if (!done) {
        max_next_q = agent->q_table[next_state][0];
        for (int a = 1; a < agent->num_actions; a++) {
            if (agent->q_table[next_state][a] > max_next_q) {
                max_next_q = agent->q_table[next_state][a];
            }
        }
    }

    // Q-learning update formula: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
    float td_target = reward + agent->discount_factor * max_next_q;
    float td_error = td_target - current_q;
    agent->q_table[state][action] = current_q + agent->learning_rate * td_error;

    // Store last action for reference
    agent->last_action = action;
}

// Decay epsilon for exploration reduction over time
void decay_epsilon(QLearningAgent* agent) {
    if (!agent) return;

    agent->epsilon *= agent->epsilon_decay;
    if (agent->epsilon < agent->epsilon_min) {
        agent->epsilon = agent->epsilon_min;
    }
}

// Get Q-value for a specific state-action pair
float get_q_value(QLearningAgent* agent, int state, Action action) {
    if (!agent || state < 0 || state >= agent->num_states || 
        (int)action < 0 || (int)action >= agent->num_actions) {
        return 0.0f;
    }
    return agent->q_table[state][action];
}

// Set Q-value for a specific state-action pair
void set_q_value(QLearningAgent* agent, int state, Action action, float value) {
    if (!agent || state < 0 || state >= agent->num_states || 
        (int)action < 0 || (int)action >= agent->num_actions) {
        return;
    }
    agent->q_table[state][action] = value;
}

// Experience buffer functions
ExperienceBuffer* create_experience_buffer(int capacity) {
    ExperienceBuffer* buffer = (ExperienceBuffer*)malloc(sizeof(ExperienceBuffer));
    if (!buffer) {
        fprintf(stderr, "Error: Failed to allocate memory for experience buffer\n");
        return NULL;
    }

    buffer->experiences = (Experience*)malloc(capacity * sizeof(Experience));
    if (!buffer->experiences) {
        fprintf(stderr, "Error: Failed to allocate memory for experiences\n");
        free(buffer);
        return NULL;
    }

    buffer->capacity = capacity;
    buffer->size = 0;
    buffer->current_index = 0;

    return buffer;
}

void destroy_experience_buffer(ExperienceBuffer* buffer) {
    if (!buffer) return;
    free(buffer->experiences);
    free(buffer);
}

void add_experience(ExperienceBuffer* buffer, int state, Action action, float reward, int next_state, bool done) {
    if (!buffer) return;

    Experience* exp = &buffer->experiences[buffer->current_index];
    exp->state = state;
    exp->action = action;
    exp->reward = reward;
    exp->next_state = next_state;
    exp->done = done;

    buffer->current_index = (buffer->current_index + 1) % buffer->capacity;
    if (buffer->size < buffer->capacity) {
        buffer->size++;
    }
}

Experience* sample_experience(ExperienceBuffer* buffer) {
    if (!buffer || buffer->size == 0) return NULL;
    
    int index = rand() % buffer->size;
    return &buffer->experiences[index];
}

// Training statistics functions
TrainingStats* create_training_stats(int max_episodes) {
    TrainingStats* stats = (TrainingStats*)malloc(sizeof(TrainingStats));
    if (!stats) {
        fprintf(stderr, "Error: Failed to allocate memory for training stats\n");
        return NULL;
    }

    stats->episodes = (EpisodeStats*)calloc(max_episodes, sizeof(EpisodeStats));
    if (!stats->episodes) {
        fprintf(stderr, "Error: Failed to allocate memory for episode stats\n");
        free(stats);
        return NULL;
    }

    stats->max_episodes = max_episodes;
    stats->current_episode = 0;
    stats->best_reward = -INFINITY;
    stats->best_episode = 0;

    return stats;
}

void destroy_training_stats(TrainingStats* stats) {
    if (!stats) return;
    free(stats->episodes);
    free(stats);
}

void record_episode(TrainingStats* stats, int episode, float total_reward, int steps_taken, float epsilon_used, float avg_q_value) {
    if (!stats || episode >= stats->max_episodes) return;

    EpisodeStats* ep_stats = &stats->episodes[episode];
    ep_stats->episode = episode;
    ep_stats->total_reward = total_reward;
    ep_stats->steps_taken = steps_taken;
    ep_stats->epsilon_used = epsilon_used;
    ep_stats->avg_q_value = avg_q_value;

    // Update best reward tracking
    if (total_reward > stats->best_reward) {
        stats->best_reward = total_reward;
        stats->best_episode = episode;
    }

    stats->current_episode = episode + 1;
}

void print_training_summary(TrainingStats* stats) {
    if (!stats) return;

    printf("\n=== Training Summary ===\n");
    printf("Total Episodes: %d\n", stats->current_episode);
    printf("Best Episode: %d (Reward: %.2f)\n", stats->best_episode, stats->best_reward);

    if (stats->current_episode > 0) {
        // Calculate average reward over all episodes
        float total_reward = 0.0f;
        int total_steps = 0;
        for (int i = 0; i < stats->current_episode; i++) {
            total_reward += stats->episodes[i].total_reward;
            total_steps += stats->episodes[i].steps_taken;
        }
        
        float avg_reward = total_reward / stats->current_episode;
        float avg_steps = (float)total_steps / stats->current_episode;
        
        printf("Average Reward: %.2f\n", avg_reward);
        printf("Average Steps per Episode: %.1f\n", avg_steps);

        // Show last few episodes
        printf("\nLast 5 Episodes:\n");
        int start_episode = (stats->current_episode > 5) ? stats->current_episode - 5 : 0;
        for (int i = start_episode; i < stats->current_episode; i++) {
            EpisodeStats* ep = &stats->episodes[i];
            printf("Episode %d: Reward=%.1f, Steps=%d, Epsilon=%.3f\n", 
                   ep->episode, ep->total_reward, ep->steps_taken, ep->epsilon_used);
        }
    }
    printf("========================\n\n");
}
