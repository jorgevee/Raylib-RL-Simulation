#include "agent.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <limits.h>

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

// Performance metrics functions
PerformanceMetrics* create_performance_metrics(int max_episodes, int window_size, int convergence_threshold) {
    PerformanceMetrics* metrics = (PerformanceMetrics*)malloc(sizeof(PerformanceMetrics));
    if (!metrics) {
        fprintf(stderr, "Error: Failed to allocate memory for performance metrics\n");
        return NULL;
    }

    metrics->moving_avg_rewards = (float*)calloc(max_episodes, sizeof(float));
    metrics->moving_avg_steps = (float*)calloc(max_episodes, sizeof(float));
    metrics->success_episodes = (int*)calloc(max_episodes, sizeof(int));
    metrics->q_value_variance = (float*)calloc(max_episodes, sizeof(float));
    metrics->epsilon_history = (float*)calloc(max_episodes, sizeof(float));

    if (!metrics->moving_avg_rewards || !metrics->moving_avg_steps || 
        !metrics->success_episodes || !metrics->q_value_variance || !metrics->epsilon_history) {
        fprintf(stderr, "Error: Failed to allocate memory for performance metrics arrays\n");
        destroy_performance_metrics(metrics);
        return NULL;
    }

    metrics->window_size = window_size;
    metrics->convergence_threshold = convergence_threshold;
    metrics->has_converged = false;
    metrics->convergence_episode = -1;

    return metrics;
}

void destroy_performance_metrics(PerformanceMetrics* metrics) {
    if (!metrics) return;
    
    free(metrics->moving_avg_rewards);
    free(metrics->moving_avg_steps);
    free(metrics->success_episodes);
    free(metrics->q_value_variance);
    free(metrics->epsilon_history);
    free(metrics);
}

float calculate_moving_average(float* values, int start, int count) {
    if (!values || count <= 0) return 0.0f;
    
    float sum = 0.0f;
    for (int i = start; i < start + count; i++) {
        sum += values[i];
    }
    return sum / count;
}

float calculate_q_value_variance(QLearningAgent* agent) {
    if (!agent) return 0.0f;
    
    // Calculate mean Q-value
    float sum = 0.0f;
    int total_entries = agent->num_states * agent->num_actions;
    
    for (int s = 0; s < agent->num_states; s++) {
        for (int a = 0; a < agent->num_actions; a++) {
            sum += agent->q_table[s][a];
        }
    }
    float mean = sum / total_entries;
    
    // Calculate variance
    float variance_sum = 0.0f;
    for (int s = 0; s < agent->num_states; s++) {
        for (int a = 0; a < agent->num_actions; a++) {
            float diff = agent->q_table[s][a] - mean;
            variance_sum += diff * diff;
        }
    }
    
    return variance_sum / total_entries;
}

void update_performance_metrics(PerformanceMetrics* metrics, TrainingStats* stats, int episode, bool goal_reached, float q_variance) {
    if (!metrics || !stats || episode >= stats->max_episodes) return;
    
    EpisodeStats* ep_stats = &stats->episodes[episode];
    
    // Store raw values
    metrics->success_episodes[episode] = goal_reached ? 1 : 0;
    metrics->q_value_variance[episode] = q_variance;
    metrics->epsilon_history[episode] = ep_stats->epsilon_used;
    
    // Calculate moving averages
    int window_start = (episode >= metrics->window_size) ? episode - metrics->window_size + 1 : 0;
    int window_count = episode - window_start + 1;
    
    // Calculate moving average of rewards
    float reward_sum = 0.0f;
    for (int i = window_start; i <= episode; i++) {
        reward_sum += stats->episodes[i].total_reward;
    }
    metrics->moving_avg_rewards[episode] = reward_sum / window_count;
    
    // Calculate moving average of steps
    float steps_sum = 0.0f;
    for (int i = window_start; i <= episode; i++) {
        steps_sum += stats->episodes[i].steps_taken;
    }
    metrics->moving_avg_steps[episode] = steps_sum / window_count;
}

bool check_convergence(PerformanceMetrics* metrics, int current_episode) {
    if (!metrics || metrics->has_converged || current_episode < metrics->convergence_threshold) {
        return metrics->has_converged;
    }
    
    // Check if reward variance is low over the last convergence_threshold episodes
    int start_episode = current_episode - metrics->convergence_threshold + 1;
    float reward_variance = 0.0f;
    float mean_reward = 0.0f;
    
    // Calculate mean reward over convergence window
    for (int i = start_episode; i <= current_episode; i++) {
        mean_reward += metrics->moving_avg_rewards[i];
    }
    mean_reward /= metrics->convergence_threshold;
    
    // Calculate variance of rewards over convergence window
    for (int i = start_episode; i <= current_episode; i++) {
        float diff = metrics->moving_avg_rewards[i] - mean_reward;
        reward_variance += diff * diff;
    }
    reward_variance /= metrics->convergence_threshold;
    
    // Check convergence criteria (low variance and high success rate)
    float success_rate = 0.0f;
    for (int i = start_episode; i <= current_episode; i++) {
        success_rate += metrics->success_episodes[i];
    }
    success_rate /= metrics->convergence_threshold;
    
    // Convergence criteria: low reward variance and high success rate
    if (reward_variance < 5.0f && success_rate > 0.8f) {
        metrics->has_converged = true;
        metrics->convergence_episode = current_episode;
        return true;
    }
    
    return false;
}

void print_learning_curves(TrainingStats* stats, int last_n_episodes) {
    if (!stats || !stats->metrics) return;
    
    printf("\n=== Learning Curves (Last %d Episodes) ===\n", last_n_episodes);
    
    int start_episode = (stats->current_episode > last_n_episodes) ? 
                       stats->current_episode - last_n_episodes : 0;
    
    printf("Episode | Reward | MovAvg | Steps | Success | Epsilon | Q-Var\n");
    printf("--------|--------|--------|-------|---------|---------|-------\n");
    
    for (int i = start_episode; i < stats->current_episode; i++) {
        EpisodeStats* ep = &stats->episodes[i];
        PerformanceMetrics* metrics = stats->metrics;
        
        printf("%7d | %6.1f | %6.1f | %5d | %7s | %7.3f | %6.2f\n",
               ep->episode + 1,
               ep->total_reward,
               metrics->moving_avg_rewards[i],
               ep->steps_taken,
               metrics->success_episodes[i] ? "Yes" : "No",
               metrics->epsilon_history[i],
               metrics->q_value_variance[i]);
    }
    printf("===============================================\n");
}

void print_convergence_analysis(PerformanceMetrics* metrics, int current_episode) {
    if (!metrics) return;
    
    printf("\n=== Convergence Analysis ===\n");
    
    if (metrics->has_converged) {
        printf("✓ CONVERGENCE DETECTED at episode %d\n", metrics->convergence_episode + 1);
    } else {
        printf("⧗ Training in progress...\n");
    }
    
    // Calculate recent performance statistics
    if (current_episode >= metrics->window_size) {
        int window_start = current_episode - metrics->window_size + 1;
        
        // Success rate over window
        float success_rate = 0.0f;
        for (int i = window_start; i <= current_episode; i++) {
            success_rate += metrics->success_episodes[i];
        }
        success_rate /= metrics->window_size;
        
        // Average performance over window
        float avg_reward = metrics->moving_avg_rewards[current_episode];
        float avg_steps = metrics->moving_avg_steps[current_episode];
        float current_q_var = metrics->q_value_variance[current_episode];
        
        printf("Recent Performance (Window size: %d):\n", metrics->window_size);
        printf("  Success Rate: %.1f%%\n", success_rate * 100.0f);
        printf("  Avg Reward: %.2f\n", avg_reward);
        printf("  Avg Steps: %.1f\n", avg_steps);
        printf("  Q-Value Variance: %.3f\n", current_q_var);
        printf("  Current Epsilon: %.3f\n", metrics->epsilon_history[current_episode]);
    }
    
    printf("=============================\n");
}

void save_performance_data(TrainingStats* stats, const char* filename) {
    if (!stats || !filename) return;
    
    FILE* file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Error: Could not create performance data file %s\n", filename);
        return;
    }
    
    fprintf(file, "# Q-Learning Performance Data\n");
    fprintf(file, "# Episode,Reward,Steps,Success,MovAvgReward,MovAvgSteps,Epsilon,QVariance\n");
    
    for (int i = 0; i < stats->current_episode; i++) {
        EpisodeStats* ep = &stats->episodes[i];
        PerformanceMetrics* metrics = stats->metrics;
        
        fprintf(file, "%d,%.2f,%d,%d,%.2f,%.2f,%.4f,%.4f\n",
                ep->episode + 1,
                ep->total_reward,
                ep->steps_taken,
                metrics->success_episodes[i],
                metrics->moving_avg_rewards[i],
                metrics->moving_avg_steps[i],
                metrics->epsilon_history[i],
                metrics->q_value_variance[i]);
    }
    
    fclose(file);
    printf("Performance data saved to %s\n", filename);
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

    // Initialize enhanced statistics
    stats->max_episodes = max_episodes;
    stats->current_episode = 0;
    stats->best_reward = -INFINITY;
    stats->best_episode = 0;
    stats->worst_reward = INFINITY;
    stats->worst_episode = 0;
    stats->total_successful_episodes = 0;
    stats->avg_reward_all_episodes = 0.0f;
    stats->avg_steps_all_episodes = 0.0f;
    
    // Create performance metrics with default parameters
    stats->metrics = create_performance_metrics(max_episodes, 100, 50); // Window size: 100, Convergence threshold: 50
    if (!stats->metrics) {
        fprintf(stderr, "Error: Failed to create performance metrics\n");
        free(stats->episodes);
        free(stats);
        return NULL;
    }

    return stats;
}

void destroy_training_stats(TrainingStats* stats) {
    if (!stats) return;
    
    if (stats->metrics) {
        destroy_performance_metrics(stats->metrics);
    }
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

// Save Q-table to binary file
bool save_q_table(QLearningAgent* agent, const char* filename) {
    if (!agent || !filename) {
        fprintf(stderr, "Error: Invalid parameters for save_q_table\n");
        return false;
    }

    FILE* file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Error: Could not open file %s for writing\n", filename);
        return false;
    }

    // Write header information
    fwrite(&agent->num_states, sizeof(int), 1, file);
    fwrite(&agent->num_actions, sizeof(int), 1, file);
    fwrite(&agent->learning_rate, sizeof(float), 1, file);
    fwrite(&agent->discount_factor, sizeof(float), 1, file);
    fwrite(&agent->epsilon, sizeof(float), 1, file);
    fwrite(&agent->epsilon_decay, sizeof(float), 1, file);
    fwrite(&agent->epsilon_min, sizeof(float), 1, file);

    // Write Q-table data
    for (int state = 0; state < agent->num_states; state++) {
        fwrite(agent->q_table[state], sizeof(float), agent->num_actions, file);
    }

    fclose(file);
    printf("Q-table saved to %s\n", filename);
    return true;
}

// Load Q-table from binary file
bool load_q_table(QLearningAgent* agent, const char* filename) {
    if (!agent || !filename) {
        fprintf(stderr, "Error: Invalid parameters for load_q_table\n");
        return false;
    }

    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: Could not open file %s for reading\n", filename);
        return false;
    }

    // Read header information
    int num_states, num_actions;
    fread(&num_states, sizeof(int), 1, file);
    fread(&num_actions, sizeof(int), 1, file);

    // Verify compatibility
    if (num_states != agent->num_states || num_actions != agent->num_actions) {
        fprintf(stderr, "Error: Q-table dimensions mismatch. Expected %dx%d, got %dx%d\n",
                agent->num_states, agent->num_actions, num_states, num_actions);
        fclose(file);
        return false;
    }

    // Load agent parameters
    fread(&agent->learning_rate, sizeof(float), 1, file);
    fread(&agent->discount_factor, sizeof(float), 1, file);
    fread(&agent->epsilon, sizeof(float), 1, file);
    fread(&agent->epsilon_decay, sizeof(float), 1, file);
    fread(&agent->epsilon_min, sizeof(float), 1, file);

    // Load Q-table data
    for (int state = 0; state < agent->num_states; state++) {
        fread(agent->q_table[state], sizeof(float), agent->num_actions, file);
    }

    fclose(file);
    printf("Q-table loaded from %s\n", filename);
    return true;
}

// ============================================================================
// PRIORITY EXPERIENCE REPLAY IMPLEMENTATION
// ============================================================================

// Create default replay configuration
ReplayConfig create_default_replay_config() {
    ReplayConfig config = {
        .enabled = true,
        .buffer_size = 10000,
        .batch_size = 32,
        .replay_frequency = 4,
        .priority_alpha = 0.6f,
        .priority_beta_start = 0.4f,
        .priority_beta_end = 1.0f,
        .beta_anneal_steps = 100000,
        .min_priority = 1e-6f
    };
    return config;
}

// Create custom replay configuration
ReplayConfig create_replay_config(bool enabled, int buffer_size, int batch_size, int replay_frequency, 
                                 float priority_alpha, float priority_beta_start, float priority_beta_end, 
                                 int beta_anneal_steps, float min_priority) {
    ReplayConfig config = {
        .enabled = enabled,
        .buffer_size = buffer_size,
        .batch_size = batch_size,
        .replay_frequency = replay_frequency,
        .priority_alpha = priority_alpha,
        .priority_beta_start = priority_beta_start,
        .priority_beta_end = priority_beta_end,
        .beta_anneal_steps = beta_anneal_steps,
        .min_priority = min_priority
    };
    return config;
}

// Priority queue helper functions
void heapify_up(PriorityExperienceBuffer* buffer, int index) {
    if (index == 0) return;
    
    int parent = (index - 1) / 2;
    int heap_idx = buffer->heap[index];
    int parent_heap_idx = buffer->heap[parent];
    
    if (buffer->priorities[heap_idx] > buffer->priorities[parent_heap_idx]) {
        // Swap heap indices
        buffer->heap[index] = parent_heap_idx;
        buffer->heap[parent] = heap_idx;
        heapify_up(buffer, parent);
    }
}

void heapify_down(PriorityExperienceBuffer* buffer, int index) {
    int left_child = 2 * index + 1;
    int right_child = 2 * index + 2;
    int largest = index;
    
    if (left_child < buffer->size) {
        int left_heap_idx = buffer->heap[left_child];
        int largest_heap_idx = buffer->heap[largest];
        if (buffer->priorities[left_heap_idx] > buffer->priorities[largest_heap_idx]) {
            largest = left_child;
        }
    }
    
    if (right_child < buffer->size) {
        int right_heap_idx = buffer->heap[right_child];
        int largest_heap_idx = buffer->heap[largest];
        if (buffer->priorities[right_heap_idx] > buffer->priorities[largest_heap_idx]) {
            largest = right_child;
        }
    }
    
    if (largest != index) {
        // Swap heap indices
        int temp = buffer->heap[index];
        buffer->heap[index] = buffer->heap[largest];
        buffer->heap[largest] = temp;
        heapify_down(buffer, largest);
    }
}

void heap_insert(PriorityExperienceBuffer* buffer, int experience_index, float priority) {
    if (buffer->size >= buffer->capacity) return;
    
    buffer->priorities[experience_index] = priority;
    buffer->heap[buffer->size] = experience_index;
    heapify_up(buffer, buffer->size);
    buffer->size++;
}

int heap_extract_max(PriorityExperienceBuffer* buffer) {
    if (buffer->size == 0) return -1;
    
    int max_idx = buffer->heap[0];
    buffer->heap[0] = buffer->heap[buffer->size - 1];
    buffer->size--;
    
    if (buffer->size > 0) {
        heapify_down(buffer, 0);
    }
    
    return max_idx;
}

// Create priority experience buffer
PriorityExperienceBuffer* create_priority_buffer(int capacity, ReplayConfig config) {
    PriorityExperienceBuffer* buffer = (PriorityExperienceBuffer*)malloc(sizeof(PriorityExperienceBuffer));
    if (!buffer) {
        fprintf(stderr, "Error: Failed to allocate memory for priority experience buffer\n");
        return NULL;
    }

    buffer->experiences = (PriorityExperience*)malloc(capacity * sizeof(PriorityExperience));
    buffer->priorities = (float*)malloc(capacity * sizeof(float));
    buffer->heap = (int*)malloc(capacity * sizeof(int));
    
    if (!buffer->experiences || !buffer->priorities || !buffer->heap) {
        fprintf(stderr, "Error: Failed to allocate memory for priority buffer components\n");
        destroy_priority_buffer(buffer);
        return NULL;
    }

    buffer->capacity = capacity;
    buffer->size = 0;
    buffer->current_index = 0;
    buffer->alpha = config.priority_alpha;
    buffer->beta = config.priority_beta_start;
    buffer->beta_increment = (config.priority_beta_end - config.priority_beta_start) / config.beta_anneal_steps;
    buffer->max_priority = 1.0f;
    buffer->min_priority = config.min_priority;
    buffer->replay_batch_size = config.batch_size;
    buffer->global_step = 0;

    // Initialize priorities to minimum value
    for (int i = 0; i < capacity; i++) {
        buffer->priorities[i] = buffer->min_priority;
    }

    return buffer;
}

// Destroy priority experience buffer
void destroy_priority_buffer(PriorityExperienceBuffer* buffer) {
    if (!buffer) return;
    
    free(buffer->experiences);
    free(buffer->priorities);
    free(buffer->heap);
    free(buffer);
}

// Add experience with priority
void add_priority_experience(PriorityExperienceBuffer* buffer, int state, Action action, float reward, 
                           int next_state, bool done, float td_error) {
    if (!buffer) return;

    PriorityExperience* exp = &buffer->experiences[buffer->current_index];
    exp->state = state;
    exp->action = action;
    exp->reward = reward;
    exp->next_state = next_state;
    exp->done = done;
    exp->td_error = td_error;
    exp->timestamp = buffer->global_step++;

    // Calculate priority from TD error
    float priority = powf(fabsf(td_error) + buffer->min_priority, buffer->alpha);
    exp->priority = priority;
    
    // Update priority in the priority array
    buffer->priorities[buffer->current_index] = priority;
    
    // Update max priority (recalculate if this is the first experience or if we're overwriting)
    if (buffer->size == 0 || priority > buffer->max_priority) {
        buffer->max_priority = priority;
    } else if (buffer->size >= buffer->capacity && buffer->current_index < buffer->size) {
        // We might be overwriting the max priority experience, recalculate max
        buffer->max_priority = buffer->priorities[0];
        for (int i = 1; i < buffer->size; i++) {
            if (buffer->priorities[i] > buffer->max_priority) {
                buffer->max_priority = buffer->priorities[i];
            }
        }
    }

    buffer->current_index = (buffer->current_index + 1) % buffer->capacity;
    if (buffer->size < buffer->capacity) {
        buffer->size++;
    }
}

// Calculate importance sampling weight
float calculate_importance_weight(PriorityExperienceBuffer* buffer, int index) {
    if (!buffer || index < 0 || index >= buffer->size) return 1.0f;
    
    float priority = buffer->priorities[index];
    float prob = priority / buffer->max_priority;  // Simplified probability calculation
    float weight = powf(prob * buffer->size, -buffer->beta);
    
    return weight;
}

// Update beta for importance sampling annealing
void update_beta(PriorityExperienceBuffer* buffer) {
    if (!buffer) return;
    
    buffer->beta = fminf(buffer->beta + buffer->beta_increment, 1.0f);
}

// Sample priority batch with importance weights
PriorityExperience* sample_priority_batch(PriorityExperienceBuffer* buffer, int batch_size, 
                                        int* indices, float* weights) {
    if (!buffer || buffer->size == 0 || batch_size <= 0) return NULL;
    
    static PriorityExperience* batch = NULL;
    static int batch_capacity = 0;
    
    // Reallocate batch if needed
    if (batch_capacity < batch_size) {
        batch = (PriorityExperience*)realloc(batch, batch_size * sizeof(PriorityExperience));
        batch_capacity = batch_size;
    }
    
    // Calculate total priority sum for sampling
    float total_priority = 0.0f;
    for (int i = 0; i < buffer->size; i++) {
        total_priority += buffer->priorities[i];
    }
    
    // Sample experiences based on priority
    for (int i = 0; i < batch_size; i++) {
        float random_value = ((float)rand() / RAND_MAX) * total_priority;
        float cumulative_priority = 0.0f;
        int selected_index = 0;
        
        for (int j = 0; j < buffer->size; j++) {
            cumulative_priority += buffer->priorities[j];
            if (cumulative_priority >= random_value) {
                selected_index = j;
                break;
            }
        }
        
        indices[i] = selected_index;
        batch[i] = buffer->experiences[selected_index];
        weights[i] = calculate_importance_weight(buffer, selected_index);
    }
    
    return batch;
}

// Update experience priorities based on new TD errors
void update_experience_priorities(PriorityExperienceBuffer* buffer, int* indices, float* td_errors, int count) {
    if (!buffer || !indices || !td_errors) return;
    
    for (int i = 0; i < count; i++) {
        int idx = indices[i];
        if (idx >= 0 && idx < buffer->size) {
            float new_priority = powf(fabsf(td_errors[i]) + buffer->min_priority, buffer->alpha);
            buffer->priorities[idx] = new_priority;
            buffer->experiences[idx].td_error = td_errors[i];
            buffer->experiences[idx].priority = new_priority;
            
            // Update max priority
            if (new_priority > buffer->max_priority) {
                buffer->max_priority = new_priority;
            }
        }
    }
}

// Calculate TD error for an experience
float calculate_td_error(QLearningAgent* agent, PriorityExperience* exp) {
    if (!agent || !exp) return 0.0f;
    
    float current_q = get_q_value(agent, exp->state, exp->action);
    float max_next_q = 0.0f;
    
    if (!exp->done) {
        // Find maximum Q-value for next state
        for (int a = 0; a < agent->num_actions; a++) {
            float q_val = get_q_value(agent, exp->next_state, a);
            if (a == 0 || q_val > max_next_q) {
                max_next_q = q_val;
            }
        }
    }
    
    float td_target = exp->reward + agent->discount_factor * max_next_q;
    return td_target - current_q;
}

// Replay batch of experiences with importance sampling
void replay_batch_experiences(QLearningAgent* agent, PriorityExperience* batch, 
                            float* importance_weights, int batch_size) {
    if (!agent || !batch || !importance_weights) return;
    
    for (int i = 0; i < batch_size; i++) {
        PriorityExperience* exp = &batch[i];
        
        // Calculate current Q-value
        float current_q = get_q_value(agent, exp->state, exp->action);
        float max_next_q = 0.0f;
        
        // Find maximum Q-value for next state (if not terminal)
        if (!exp->done) {
            for (int a = 0; a < agent->num_actions; a++) {
                float q_val = get_q_value(agent, exp->next_state, a);
                if (a == 0 || q_val > max_next_q) {
                    max_next_q = q_val;
                }
            }
        }
        
        // Calculate TD target and error
        float td_target = exp->reward + agent->discount_factor * max_next_q;
        float td_error = td_target - current_q;
        
        // Apply importance sampling weight to learning rate
        float weighted_lr = agent->learning_rate * importance_weights[i];
        
        // Update Q-value
        float new_q = current_q + weighted_lr * td_error;
        set_q_value(agent, exp->state, exp->action, new_q);
    }
}

// ============================================================================
// STATE VISIT TRACKING IMPLEMENTATION
// ============================================================================

// Create state visit tracker
StateVisitTracker* create_state_visit_tracker(int num_states, bool adaptive_epsilon, bool adaptive_learning_rate) {
    StateVisitTracker* tracker = (StateVisitTracker*)malloc(sizeof(StateVisitTracker));
    if (!tracker) {
        fprintf(stderr, "Error: Failed to allocate memory for state visit tracker\n");
        return NULL;
    }

    tracker->visit_counts = (int*)calloc(num_states, sizeof(int));
    tracker->visit_priorities = (float*)calloc(num_states, sizeof(float));
    tracker->exploration_bonuses = (float*)malloc(num_states * sizeof(float));
    tracker->state_epsilons = (float*)malloc(num_states * sizeof(float));
    tracker->state_learning_rates = (float*)malloc(num_states * sizeof(float));

    if (!tracker->visit_counts || !tracker->visit_priorities || !tracker->exploration_bonuses ||
        !tracker->state_epsilons || !tracker->state_learning_rates) {
        fprintf(stderr, "Error: Failed to allocate memory for state visit tracker arrays\n");
        destroy_state_visit_tracker(tracker);
        return NULL;
    }

    tracker->num_states = num_states;
    tracker->total_visits = 0;
    tracker->exploration_bonus_decay = 0.999f;
    tracker->min_exploration_bonus = 0.01f;
    tracker->priority_temperature = 1.0f;
    tracker->adaptive_epsilon = adaptive_epsilon;
    tracker->adaptive_learning_rate = adaptive_learning_rate;

    // Initialize arrays
    for (int i = 0; i < num_states; i++) {
        tracker->exploration_bonuses[i] = 1.0f;    // Start with high exploration bonus
        tracker->state_epsilons[i] = 1.0f;        // Start with high exploration
        tracker->state_learning_rates[i] = 1.0f;  // Start with normal learning rate
        tracker->visit_priorities[i] = 1.0f;      // Equal priority initially
    }

    return tracker;
}

// Destroy state visit tracker
void destroy_state_visit_tracker(StateVisitTracker* tracker) {
    if (!tracker) return;
    
    free(tracker->visit_counts);
    free(tracker->visit_priorities);
    free(tracker->exploration_bonuses);
    free(tracker->state_epsilons);
    free(tracker->state_learning_rates);
    free(tracker);
}

// Update state visit count and derived metrics
void update_state_visit(StateVisitTracker* tracker, int state) {
    if (!tracker || state < 0 || state >= tracker->num_states) return;
    
    tracker->visit_counts[state]++;
    tracker->total_visits++;
    
    // Update exploration bonus (decreases with visits)
    tracker->exploration_bonuses[state] = fmaxf(
        tracker->min_exploration_bonus,
        1.0f / sqrtf((float)tracker->visit_counts[state] + 1)
    );
    
    // Update state-specific epsilon (less exploration for well-visited states)
    if (tracker->adaptive_epsilon) {
        tracker->state_epsilons[state] = tracker->exploration_bonuses[state];
    }
    
    // Update state-specific learning rate (faster learning in new states)
    if (tracker->adaptive_learning_rate) {
        tracker->state_learning_rates[state] = fminf(
            2.0f,
            1.0f + tracker->exploration_bonuses[state]
        );
    }
    
    // Update visit priorities
    update_state_priorities(tracker);
}

// Get exploration bonus for a state
float get_exploration_bonus(StateVisitTracker* tracker, int state) {
    if (!tracker || state < 0 || state >= tracker->num_states) return 0.0f;
    
    return tracker->exploration_bonuses[state];
}

// Get adaptive epsilon for a state
float get_state_epsilon(StateVisitTracker* tracker, int state, float base_epsilon) {
    if (!tracker || state < 0 || state >= tracker->num_states || !tracker->adaptive_epsilon) {
        return base_epsilon;
    }
    
    return base_epsilon * tracker->state_epsilons[state];
}

// Get adaptive learning rate for a state
float get_state_learning_rate(StateVisitTracker* tracker, int state, float base_learning_rate) {
    if (!tracker || state < 0 || state >= tracker->num_states || !tracker->adaptive_learning_rate) {
        return base_learning_rate;
    }
    
    return base_learning_rate * tracker->state_learning_rates[state];
}

// Decay exploration bonuses over time
void decay_exploration_bonuses(StateVisitTracker* tracker) {
    if (!tracker) return;
    
    for (int i = 0; i < tracker->num_states; i++) {
        tracker->exploration_bonuses[i] *= tracker->exploration_bonus_decay;
        tracker->exploration_bonuses[i] = fmaxf(
            tracker->exploration_bonuses[i], 
            tracker->min_exploration_bonus
        );
    }
}

// Select state with highest priority (least visited or highest exploration bonus)
int select_priority_state(StateVisitTracker* tracker) {
    if (!tracker) return 0;
    
    int best_state = 0;
    float best_priority = tracker->visit_priorities[0];
    
    for (int i = 1; i < tracker->num_states; i++) {
        if (tracker->visit_priorities[i] > best_priority) {
            best_priority = tracker->visit_priorities[i];
            best_state = i;
        }
    }
    
    return best_state;
}

// Update state priorities based on visit counts and exploration bonuses
void update_state_priorities(StateVisitTracker* tracker) {
    if (!tracker) return;
    
    // Find min and max visit counts for normalization
    int min_visits = tracker->visit_counts[0];
    int max_visits = tracker->visit_counts[0];
    
    for (int i = 1; i < tracker->num_states; i++) {
        if (tracker->visit_counts[i] < min_visits) min_visits = tracker->visit_counts[i];
        if (tracker->visit_counts[i] > max_visits) max_visits = tracker->visit_counts[i];
    }
    
    // Calculate priorities (higher for less visited states)
    for (int i = 0; i < tracker->num_states; i++) {
        if (max_visits == min_visits) {
            tracker->visit_priorities[i] = 1.0f;  // Equal priority if all same
        } else {
            // Inverse relationship: fewer visits = higher priority
            float visit_norm = 1.0f - ((float)(tracker->visit_counts[i] - min_visits) / (max_visits - min_visits));
            tracker->visit_priorities[i] = visit_norm + tracker->exploration_bonuses[i];
        }
    }
}

// Reset state visit tracker
void reset_state_visit_tracker(StateVisitTracker* tracker) {
    if (!tracker) return;
    
    // Reset all counters and arrays
    memset(tracker->visit_counts, 0, tracker->num_states * sizeof(int));
    tracker->total_visits = 0;
    
    // Reset to initial values
    for (int i = 0; i < tracker->num_states; i++) {
        tracker->exploration_bonuses[i] = 1.0f;
        tracker->state_epsilons[i] = 1.0f;
        tracker->state_learning_rates[i] = 1.0f;
        tracker->visit_priorities[i] = 1.0f;
    }
}

// Enhanced action selection with state visit priority
Action select_action_with_priority(QLearningAgent* agent, StateVisitTracker* tracker, int state) {
    if (!agent || state < 0 || state >= agent->num_states) {
        return ACTION_UP; // Default action
    }

    agent->current_state = state;
    
    // Update state visit
    if (tracker) {
        update_state_visit(tracker, state);
    }

    // Get adaptive epsilon for this state
    float epsilon = agent->epsilon;
    if (tracker && tracker->adaptive_epsilon) {
        epsilon = get_state_epsilon(tracker, state, agent->epsilon);
    }

    // Epsilon-greedy action selection with state-adaptive epsilon
    float random_value = (float)rand() / RAND_MAX;
    
    if (random_value < epsilon) {
        // Explore: choose random action (higher chance in less-visited states)
        return (Action)(rand() % agent->num_actions);
    } else {
        // Exploit: choose greedy action
        return select_greedy_action(agent, state);
    }
}

// Enhanced Q-value update with state visit priority
void update_q_value_with_priority(QLearningAgent* agent, StateVisitTracker* tracker, int state, 
                                 Action action, float reward, int next_state, bool done) {
    if (!agent || state < 0 || state >= agent->num_states || 
        next_state < 0 || next_state >= agent->num_states ||
        (int)action < 0 || (int)action >= agent->num_actions) {
        return;
    }

    // Get adaptive learning rate for this state
    float learning_rate = agent->learning_rate;
    if (tracker && tracker->adaptive_learning_rate) {
        learning_rate = get_state_learning_rate(tracker, state, agent->learning_rate);
    }

    // Add exploration bonus to reward for less-visited states
    float enhanced_reward = reward;
    if (tracker) {
        enhanced_reward += get_exploration_bonus(tracker, state);
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

    // Q-learning update formula with adaptive learning rate and exploration bonus
    float td_target = enhanced_reward + agent->discount_factor * max_next_q;
    float td_error = td_target - current_q;
    agent->q_table[state][action] = current_q + learning_rate * td_error;

    // Store last action for reference
    agent->last_action = action;
}

// ============================================================================
// STATE VISIT ANALYSIS AND REPORTING
// ============================================================================

// Print comprehensive state visit analysis
void print_state_visit_analysis(StateVisitTracker* tracker) {
    if (!tracker) return;
    
    printf("\n=== State Visit Analysis ===\n");
    printf("Total visits across all states: %d\n", tracker->total_visits);
    printf("Number of states: %d\n", tracker->num_states);
    
    // Calculate coverage statistics
    int visited_states = 0;
    int unvisited_states = 0;
    int min_visits = INT_MAX;
    int max_visits = 0;
    float total_exploration_bonus = 0.0f;
    
    for (int i = 0; i < tracker->num_states; i++) {
        if (tracker->visit_counts[i] > 0) {
            visited_states++;
            if (tracker->visit_counts[i] < min_visits) min_visits = tracker->visit_counts[i];
        } else {
            unvisited_states++;
        }
        if (tracker->visit_counts[i] > max_visits) max_visits = tracker->visit_counts[i];
        total_exploration_bonus += tracker->exploration_bonuses[i];
    }
    
    if (visited_states == 0) min_visits = 0;
    
    printf("Coverage Statistics:\n");
    printf("  Visited states: %d (%.1f%%)\n", visited_states, 
           (float)visited_states / tracker->num_states * 100.0f);
    printf("  Unvisited states: %d (%.1f%%)\n", unvisited_states,
           (float)unvisited_states / tracker->num_states * 100.0f);
    printf("  Min visits per state: %d\n", min_visits);
    printf("  Max visits per state: %d\n", max_visits);
    printf("  Average visits per state: %.1f\n", 
           (float)tracker->total_visits / tracker->num_states);
    printf("  Average exploration bonus: %.3f\n", 
           total_exploration_bonus / tracker->num_states);
    
    // Find and display extreme states
    int least_visited = get_least_visited_state(tracker);
    int most_visited = get_most_visited_state(tracker);
    int highest_priority = select_priority_state(tracker);
    
    printf("\nState Extremes:\n");
    printf("  Least visited state: %d (%d visits, bonus: %.3f)\n", 
           least_visited, tracker->visit_counts[least_visited], 
           tracker->exploration_bonuses[least_visited]);
    printf("  Most visited state: %d (%d visits, bonus: %.3f)\n", 
           most_visited, tracker->visit_counts[most_visited], 
           tracker->exploration_bonuses[most_visited]);
    printf("  Highest priority state: %d (priority: %.3f)\n", 
           highest_priority, tracker->visit_priorities[highest_priority]);
    
    // Configuration information
    printf("\nConfiguration:\n");
    printf("  Adaptive epsilon: %s\n", tracker->adaptive_epsilon ? "enabled" : "disabled");
    printf("  Adaptive learning rate: %s\n", tracker->adaptive_learning_rate ? "enabled" : "disabled");
    printf("  Exploration bonus decay: %.4f\n", tracker->exploration_bonus_decay);
    printf("  Min exploration bonus: %.4f\n", tracker->min_exploration_bonus);
    
    printf("=============================\n");
}

// Save state visit data to CSV format
void save_state_visit_data(StateVisitTracker* tracker, const char* filename) {
    if (!tracker || !filename) return;
    
    FILE* file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Error: Could not create state visit data file %s\n", filename);
        return;
    }
    
    fprintf(file, "# State Visit Tracking Data\n");
    fprintf(file, "# State,Visits,Priority,ExplorationBonus,StateEpsilon,StateLearningRate\n");
    
    for (int i = 0; i < tracker->num_states; i++) {
        fprintf(file, "%d,%d,%.4f,%.4f,%.4f,%.4f\n",
                i, tracker->visit_counts[i], tracker->visit_priorities[i],
                tracker->exploration_bonuses[i], tracker->state_epsilons[i],
                tracker->state_learning_rates[i]);
    }
    
    fclose(file);
    printf("State visit data saved to %s\n", filename);
}

// Calculate exploration coverage percentage
float calculate_exploration_coverage(StateVisitTracker* tracker) {
    if (!tracker) return 0.0f;
    
    int visited_states = 0;
    for (int i = 0; i < tracker->num_states; i++) {
        if (tracker->visit_counts[i] > 0) {
            visited_states++;
        }
    }
    
    return (float)visited_states / tracker->num_states * 100.0f;
}

// Get least visited state
int get_least_visited_state(StateVisitTracker* tracker) {
    if (!tracker) return 0;
    
    int least_visited = 0;
    int min_visits = tracker->visit_counts[0];
    
    for (int i = 1; i < tracker->num_states; i++) {
        if (tracker->visit_counts[i] < min_visits) {
            min_visits = tracker->visit_counts[i];
            least_visited = i;
        }
    }
    
    return least_visited;
}

// Get most visited state
int get_most_visited_state(StateVisitTracker* tracker) {
    if (!tracker) return 0;
    
    int most_visited = 0;
    int max_visits = tracker->visit_counts[0];
    
    for (int i = 1; i < tracker->num_states; i++) {
        if (tracker->visit_counts[i] > max_visits) {
            max_visits = tracker->visit_counts[i];
            most_visited = i;
        }
    }
    
    return most_visited;
}
