#ifndef UTILS_H
#define UTILS_H

#include <stdbool.h>
#include <stdio.h>
#include <time.h>

// Math constants and utilities
#define PI 3.14159265359f
#define EPSILON 1e-6f
#define MAX_PATH_LENGTH 256
#define MAX_FILENAME_LENGTH 128

// Random number generation
typedef struct {
    unsigned int seed;
    bool initialized;
} RandomState;

// File I/O structures
typedef struct {
    char filename[MAX_FILENAME_LENGTH];
    FILE* file;
    bool is_open;
    bool is_writing;
} FileHandle;

// Timer for performance measurement
typedef struct {
    clock_t start_time;
    clock_t end_time;
    double elapsed_seconds;
    bool is_running;
} Timer;

// Memory management utilities
typedef struct {
    void** allocations;
    size_t* sizes;
    int count;
    int capacity;
    size_t total_allocated;
} MemoryTracker;

// Configuration file parser
typedef struct {
    char key[64];
    char value[256];
} ConfigPair;

typedef struct {
    ConfigPair* pairs;
    int count;
    int capacity;
} ConfigFile;

// Logging system
typedef enum {
    LOG_DEBUG = 0,
    LOG_INFO = 1,
    LOG_WARNING = 2,
    LOG_ERROR = 3
} LogLevel;

typedef struct {
    FILE* log_file;
    LogLevel min_level;
    bool console_output;
    bool timestamp_enabled;
    char log_filename[MAX_FILENAME_LENGTH];
} Logger;

// Data export/import structures
typedef struct {
    float* values;
    char* labels;
    int count;
    int capacity;
} DataSeries;

typedef struct {
    DataSeries* series;
    int series_count;
    char title[128];
    char x_label[64];
    char y_label[64];
} Dataset;

// Function declarations for random number generation
RandomState* init_random(unsigned int seed);
void destroy_random(RandomState* rng);
float random_float(RandomState* rng);
float random_range(RandomState* rng, float min, float max);
int random_int(RandomState* rng, int min, int max);
bool random_bool(RandomState* rng, float probability);
void shuffle_array(RandomState* rng, void* array, size_t element_size, size_t count);

// Math utility functions
float clamp(float value, float min, float max);
float lerp(float a, float b, float t);
float map_range(float value, float in_min, float in_max, float out_min, float out_max);
int min_int(int a, int b);
int max_int(int a, int b);
float min_float(float a, float b);
float max_float(float a, float b);
float sign(float value);
bool approximately_equal(float a, float b, float tolerance);

// Array utilities
float* create_float_array(int size);
int* create_int_array(int size);
void destroy_float_array(float* array);
void destroy_int_array(int* array);
void fill_float_array(float* array, int size, float value);
void fill_int_array(int* array, int size, int value);
float array_mean(float* array, int size);
float array_std(float* array, int size);
float array_min(float* array, int size);
float array_max(float* array, int size);

// String utilities
char* string_copy(const char* source);
bool string_equals(const char* a, const char* b);
bool string_starts_with(const char* str, const char* prefix);
bool string_ends_with(const char* str, const char* suffix);
char* string_format(const char* format, ...);
void string_to_lower(char* str);
void string_to_upper(char* str);

// File I/O utilities
FileHandle* open_file(const char* filename, const char* mode);
void close_file(FileHandle* handle);
bool file_exists(const char* filename);
bool create_directory(const char* path);
char* read_entire_file(const char* filename);
bool write_text_file(const char* filename, const char* content);

// Timer functions
Timer* create_timer(void);
void destroy_timer(Timer* timer);
void start_timer(Timer* timer);
void stop_timer(Timer* timer);
double get_elapsed_time(Timer* timer);
void reset_timer(Timer* timer);

// Memory tracking (for debugging)
MemoryTracker* create_memory_tracker(void);
void destroy_memory_tracker(MemoryTracker* tracker);
void* tracked_malloc(MemoryTracker* tracker, size_t size);
void* tracked_calloc(MemoryTracker* tracker, size_t count, size_t size);
void tracked_free(MemoryTracker* tracker, void* ptr);
void print_memory_report(MemoryTracker* tracker);

// Configuration file handling
ConfigFile* load_config(const char* filename);
void destroy_config(ConfigFile* config);
const char* get_config_value(ConfigFile* config, const char* key);
int get_config_int(ConfigFile* config, const char* key, int default_value);
float get_config_float(ConfigFile* config, const char* key, float default_value);
bool get_config_bool(ConfigFile* config, const char* key, bool default_value);
void set_config_value(ConfigFile* config, const char* key, const char* value);
bool save_config(ConfigFile* config, const char* filename);

// Logging functions
Logger* create_logger(const char* filename, LogLevel min_level);
void destroy_logger(Logger* logger);
void log_message(Logger* logger, LogLevel level, const char* format, ...);
void log_debug(Logger* logger, const char* format, ...);
void log_info(Logger* logger, const char* format, ...);
void log_warning(Logger* logger, const char* format, ...);
void log_error(Logger* logger, const char* format, ...);

// Data export functions
DataSeries* create_data_series(int capacity);
void destroy_data_series(DataSeries* series);
void add_data_point(DataSeries* series, float value, const char* label);
Dataset* create_dataset(const char* title, const char* x_label, const char* y_label);
void destroy_dataset(Dataset* dataset);
void add_series_to_dataset(Dataset* dataset, DataSeries* series);
bool export_dataset_csv(Dataset* dataset, const char* filename);
bool export_dataset_json(Dataset* dataset, const char* filename);

// System utilities
double get_current_time_seconds(void);
void sleep_milliseconds(int milliseconds);
int get_cpu_count(void);
size_t get_memory_usage(void);
const char* get_platform_name(void);

// Error handling macros
#define ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            fprintf(stderr, "Assertion failed: %s at %s:%d\n", message, __FILE__, __LINE__); \
            exit(1); \
        } \
    } while(0)

#define CHECK_NULL(ptr, message) \
    do { \
        if ((ptr) == NULL) { \
            fprintf(stderr, "Null pointer: %s at %s:%d\n", message, __FILE__, __LINE__); \
            return NULL; \
        } \
    } while(0)

#define SAFE_FREE(ptr) \
    do { \
        if ((ptr) != NULL) { \
            free(ptr); \
            (ptr) = NULL; \
        } \
    } while(0)

#endif // UTILS_H
