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

// Enhanced experience with priority support
typedef struct {
    int state;
    Action action;
    float reward;
    int next_state;
    bool done;
    float td_error;           // For priority calculation
    float priority;           // Sampling priority
    int timestamp;            // For age-based prioritization
} PriorityExperience;

// Experience buffer for storing experiences
typedef struct {
    Experience* experiences;
    int capacity;
    int size;
    int current_index;
} ExperienceBuffer;

// State visit tracking for priority exploration
typedef struct {
    int* visit_counts;           // Number of times each state has been visited
    float* visit_priorities;     // Priority scores for state visits
    float* exploration_bonuses;  // Exploration bonus for each state
    float* state_epsilons;       // Adaptive epsilon per state
    float* state_learning_rates; // Adaptive learning rate per state
    int num_states;              // Total number of states
    int total_visits;            // Total state visits across all states
    float exploration_bonus_decay; // Decay rate for exploration bonuses
    float min_exploration_bonus; // Minimum exploration bonus
    float priority_temperature;  // Temperature for priority softmax
    bool adaptive_epsilon;       // Enable adaptive epsilon per state
    bool adaptive_learning_rate; // Enable adaptive learning rate per state
} StateVisitTracker;

// Priority-based experience buffer with heap-based sampling
typedef struct {
    PriorityExperience* experiences;
    float* priorities;        // Priority values for efficient access
    int* heap;               // Priority queue heap indices
    int capacity;
    int size;
    int current_index;
    float alpha;             // Priority exponent (0 = uniform, 1 = full priority)
    float beta;              // Importance sampling exponent (anneals to 1.0)
    float beta_increment;    // Beta annealing rate
    float max_priority;      // Maximum priority seen (for new experiences)
    float min_priority;      // Minimum priority to prevent zero sampling
    int replay_batch_size;   // Batch size for replay
    int global_step;         // Global step counter for timestamps
} PriorityExperienceBuffer;

// Replay configuration
typedef struct {
    bool enabled;               // Enable/disable experience replay
    int buffer_size;           // Size of experience buffer
    int batch_size;            // Batch size for replay
    int replay_frequency;      // How often to perform replay (every N steps)
    float priority_alpha;      // Priority exponent
    float priority_beta_start; // Initial importance sampling exponent
    float priority_beta_end;   // Final importance sampling exponent
    int beta_anneal_steps;     // Steps to anneal beta from start to end
    float min_priority;        // Minimum priority value
} ReplayConfig;

// Episode statistics for tracking performance
typedef struct {
    int episode;
    float total_reward;
    int steps_taken;
    float epsilon_used;
    float avg_q_value;
} EpisodeStats;

// Performance metrics for convergence analysis
typedef struct {
    float* moving_avg_rewards;    // Moving average of rewards
    float* moving_avg_steps;      // Moving average of steps
    int* success_episodes;        // Episodes where goal was reached
    float* q_value_variance;      // Variance in Q-values over time
    float* epsilon_history;       // Epsilon values over time
    int window_size;              // Window size for moving averages
    int convergence_threshold;    // Episodes to check for convergence
    bool has_converged;           // Whether training has converged
    int convergence_episode;      // Episode where convergence was detected
} PerformanceMetrics;

// Training statistics structure
typedef struct {
    EpisodeStats* episodes;
    int max_episodes;
    int current_episode;
    float best_reward;
    int best_episode;
    int worst_episode;
    float worst_reward;
    int total_successful_episodes;
    float avg_reward_all_episodes;
    float avg_steps_all_episodes;
    PerformanceMetrics* metrics;  // Advanced performance tracking
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

// Priority experience replay functions
PriorityExperienceBuffer* create_priority_buffer(int capacity, ReplayConfig config);
void destroy_priority_buffer(PriorityExperienceBuffer* buffer);
void add_priority_experience(PriorityExperienceBuffer* buffer, int state, Action action, float reward, int next_state, bool done, float td_error);
PriorityExperience* sample_priority_batch(PriorityExperienceBuffer* buffer, int batch_size, int* indices, float* weights);
void update_experience_priorities(PriorityExperienceBuffer* buffer, int* indices, float* td_errors, int count);
float calculate_importance_weight(PriorityExperienceBuffer* buffer, int index);
void update_beta(PriorityExperienceBuffer* buffer);

// Batch replay processing
void replay_batch_experiences(QLearningAgent* agent, PriorityExperience* batch, float* importance_weights, int batch_size);
float calculate_td_error(QLearningAgent* agent, PriorityExperience* exp);

// Replay configuration helpers
ReplayConfig create_default_replay_config();
ReplayConfig create_replay_config(bool enabled, int buffer_size, int batch_size, int replay_frequency, 
                                 float priority_alpha, float priority_beta_start, float priority_beta_end, 
                                 int beta_anneal_steps, float min_priority);

// Priority queue helpers (internal)
void heapify_up(PriorityExperienceBuffer* buffer, int index);
void heapify_down(PriorityExperienceBuffer* buffer, int index);
int heap_extract_max(PriorityExperienceBuffer* buffer);
void heap_insert(PriorityExperienceBuffer* buffer, int experience_index, float priority);

// Training statistics functions
TrainingStats* create_training_stats(int max_episodes);
void destroy_training_stats(TrainingStats* stats);
void record_episode(TrainingStats* stats, int episode, float total_reward, int steps_taken, float epsilon_used, float avg_q_value);
void print_training_summary(TrainingStats* stats);

// Performance metrics functions
PerformanceMetrics* create_performance_metrics(int max_episodes, int window_size, int convergence_threshold);
void destroy_performance_metrics(PerformanceMetrics* metrics);
void update_performance_metrics(PerformanceMetrics* metrics, TrainingStats* stats, int episode, bool goal_reached, float q_variance);
bool check_convergence(PerformanceMetrics* metrics, int current_episode);
float calculate_moving_average(float* values, int start, int count);
float calculate_q_value_variance(QLearningAgent* agent);
void print_learning_curves(TrainingStats* stats, int last_n_episodes);
void print_convergence_analysis(PerformanceMetrics* metrics, int current_episode);
void save_performance_data(TrainingStats* stats, const char* filename);

// Q-table save/load functions
bool save_q_table(QLearningAgent* agent, const char* filename);
bool load_q_table(QLearningAgent* agent, const char* filename);

// State visit tracking functions
StateVisitTracker* create_state_visit_tracker(int num_states, bool adaptive_epsilon, bool adaptive_learning_rate);
void destroy_state_visit_tracker(StateVisitTracker* tracker);
void update_state_visit(StateVisitTracker* tracker, int state);
float get_exploration_bonus(StateVisitTracker* tracker, int state);
float get_state_epsilon(StateVisitTracker* tracker, int state, float base_epsilon);
float get_state_learning_rate(StateVisitTracker* tracker, int state, float base_learning_rate);
void decay_exploration_bonuses(StateVisitTracker* tracker);
int select_priority_state(StateVisitTracker* tracker);
void update_state_priorities(StateVisitTracker* tracker);
void reset_state_visit_tracker(StateVisitTracker* tracker);

// Enhanced action selection with state visit priority
Action select_action_with_priority(QLearningAgent* agent, StateVisitTracker* tracker, int state);
void update_q_value_with_priority(QLearningAgent* agent, StateVisitTracker* tracker, int state, 
                                 Action action, float reward, int next_state, bool done);

// State visit analysis and reporting
void print_state_visit_analysis(StateVisitTracker* tracker);
void save_state_visit_data(StateVisitTracker* tracker, const char* filename);
float calculate_exploration_coverage(StateVisitTracker* tracker);
int get_least_visited_state(StateVisitTracker* tracker);
int get_most_visited_state(StateVisitTracker* tracker);

#endif // AGENT_H
