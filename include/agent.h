#ifndef AGENT_H
#define AGENT_H

#include <stdbool.h>

// Action enumeration for the agent
typedef enum {
    ACTION_UP = 0,
    ACTION_DOWN = 1,
    ACTION_LEFT = 2,
    ACTION_RIGHT = 3,
    NUM_ACTIONS = 4
} Action;

// Action-value pair for Q-learning
typedef struct {
    Action action;
    float q_value;
} ActionValue;

// Q-Learning Agent structure
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

// Experience structure for experience replay
typedef struct {
    int state;
    Action action;
    float reward;
    int next_state;
    bool done;
} Experience;

// Experience buffer for storing experiences
typedef struct {
    Experience* experiences;
    int capacity;
    int size;
    int current_index;
} ExperienceBuffer;

// Episode statistics for tracking performance
typedef struct {
    int episode;
    float total_reward;
    int steps_taken;
    float epsilon_used;
    float avg_q_value;
} EpisodeStats;

// Training statistics structure
typedef struct {
    EpisodeStats* episodes;
    int max_episodes;
    int current_episode;
    float best_reward;
    int best_episode;
} TrainingStats;

// Function declarations
QLearningAgent* create_agent(int num_states, int num_actions, float learning_rate, float discount_factor, float epsilon);
void destroy_agent(QLearningAgent* agent);
Action select_action(QLearningAgent* agent, int state);
Action select_greedy_action(QLearningAgent* agent, int state);
void update_q_value(QLearningAgent* agent, int state, Action action, float reward, int next_state, bool done);
void decay_epsilon(QLearningAgent* agent);
float get_q_value(QLearningAgent* agent, int state, Action action);
void set_q_value(QLearningAgent* agent, int state, Action action, float value);

// Experience buffer functions
ExperienceBuffer* create_experience_buffer(int capacity);
void destroy_experience_buffer(ExperienceBuffer* buffer);
void add_experience(ExperienceBuffer* buffer, int state, Action action, float reward, int next_state, bool done);
Experience* sample_experience(ExperienceBuffer* buffer);

// Training statistics functions
TrainingStats* create_training_stats(int max_episodes);
void destroy_training_stats(TrainingStats* stats);
void record_episode(TrainingStats* stats, int episode, float total_reward, int steps_taken, float epsilon_used, float avg_q_value);
void print_training_summary(TrainingStats* stats);

#endif // AGENT_H
