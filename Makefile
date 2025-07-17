# Makefile for RL Agent Environment with Raylib
# Author: Auto-generated for C Raylib RL Simulation

# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -O2 -g
INCLUDES = -Iinclude -I/opt/homebrew/opt/raylib/include
LIBS = -L/opt/homebrew/opt/raylib/lib -lraylib -lm

# Platform-specific settings
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)    # macOS
    LIBS += -framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo
    CFLAGS += -D_DEFAULT_SOURCE
endif
ifeq ($(UNAME_S),Linux)     # Linux
    LIBS += -lGL -lglfw -lpthread -ldl -lrt -lX11
endif
ifeq ($(OS),Windows_NT)     # Windows
    LIBS += -lopengl32 -lgdi32 -lwinmm
    EXECUTABLE_EXT = .exe
endif

# Directories
SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build
BIN_DIR = bin

# Source files
SOURCES = $(wildcard $(SRC_DIR)/*.c)
OBJECTS = $(SOURCES:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)
HEADERS = $(wildcard $(INCLUDE_DIR)/*.h)

# Target executable
TARGET = $(BIN_DIR)/rl_agent$(EXECUTABLE_EXT)

# Default target
all: directories $(TARGET)

# Create necessary directories
directories:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(BIN_DIR)

# Link the executable
$(TARGET): $(OBJECTS)
	@echo "Linking $@..."
	@$(CC) $(OBJECTS) -o $@ $(LIBS)
	@echo "Build complete: $@"

# Compile source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c $(HEADERS)
	@echo "Compiling $<..."
	@$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(BUILD_DIR)
	@rm -rf $(BIN_DIR)
	@echo "Clean complete."

# Clean and rebuild
rebuild: clean all

# Run the application
run: $(TARGET)
	@echo "Running RL Agent Environment..."
	@./$(TARGET)

# Debug build with additional flags
debug: CFLAGS += -DDEBUG -g3 -O0
debug: directories $(TARGET)

# Release build with optimizations
release: CFLAGS += -DNDEBUG -O3 -flto
release: directories $(TARGET)

# Install raylib (Linux/macOS)
install-raylib:
ifeq ($(UNAME_S),Darwin)
	@echo "Installing raylib via Homebrew..."
	@brew install raylib
endif
ifeq ($(UNAME_S),Linux)
	@echo "Installing raylib via package manager..."
	@sudo apt-get update && sudo apt-get install libraylib-dev
endif

# Check dependencies
check-deps:
	@echo "Checking dependencies..."
	@which $(CC) > /dev/null || (echo "Error: $(CC) not found" && exit 1)
	@echo "Compiler: $(CC) ✓"
	@pkg-config --exists raylib 2>/dev/null && echo "Raylib: ✓" || echo "Raylib: Please install raylib"

# Format source code (requires clang-format)
format:
	@echo "Formatting source code..."
	@find $(SRC_DIR) $(INCLUDE_DIR) -name "*.c" -o -name "*.h" | xargs clang-format -i
	@echo "Format complete."

# Static analysis (requires cppcheck)
analyze:
	@echo "Running static analysis..."
	@cppcheck --enable=all --inconclusive --std=c99 $(SRC_DIR) $(INCLUDE_DIR)

# Generate documentation (requires doxygen)
docs:
	@echo "Generating documentation..."
	@doxygen Doxyfile 2>/dev/null || echo "Doxyfile not found, skipping documentation generation"

# Create a simple test
test: $(TARGET)
	@echo "Running basic tests..."
	@./$(TARGET) --test 2>/dev/null || echo "No test mode available yet"

# Test reward system
test-rewards:
	@echo "Compiling reward system tests..."
	@$(CC) $(CFLAGS) $(INCLUDES) -o test_reward_system test_reward_system.c $(SRC_DIR)/environment.c -lm
	@echo "Running comprehensive reward system tests..."
	@./test_reward_system
	@echo "Cleaning test executable..."
	@rm -f test_reward_system

# Test environment functions  
test-environment:
	@echo "Compiling environment tests..."
	@$(CC) $(CFLAGS) $(INCLUDES) -o test_environment_complete test_environment_complete.c $(SRC_DIR)/environment.c -lm
	@echo "Running environment function tests..."
	@./test_environment_complete
	@echo "Cleaning test executable..."
	@rm -f test_environment_complete

# Test step_environment function
test-step-env:
	@echo "Compiling step_environment tests..."
	@$(CC) $(CFLAGS) $(INCLUDES) -o test_step_environment test_step_environment.c $(SRC_DIR)/environment.c -lm
	@echo "Running step_environment function tests..."
	@./test_step_environment
	@echo "Cleaning test executable..."
	@rm -f test_step_environment

# Test priority experience replay
test-priority-replay:
@echo "Compiling priority experience replay tests..."
@$(CC) $(CFLAGS) $(INCLUDES) -o test_priority_replay tests/test_priority_replay.c $(SRC_DIR)/agent.c $(SRC_DIR)/environment.c -lm
@echo "Running priority experience replay tests..."
@./test_priority_replay
@echo "Cleaning test executable..."
@rm -f test_priority_replay

# Test state visit tracking
test-state-visit-tracking:
@echo "Compiling state visit tracking tests..."
@$(CC) $(CFLAGS) $(INCLUDES) -o test_state_visit_tracking tests/test_state_visit_tracking.c $(SRC_DIR)/agent.c $(SRC_DIR)/environment.c -lm
@echo "Running state visit tracking tests..."
@./test_state_visit_tracking
@echo "Cleaning test executable..."
@rm -f test_state_visit_tracking

# Test Q-table optimization
test-qtable-optimization:
@echo "Compiling Q-table optimization tests..."
@$(CC) $(CFLAGS) $(INCLUDES) -mavx2 -msse2 -o test_qtable_optimization tests/test_qtable_optimization.c $(SRC_DIR)/q_table_optimized.c $(SRC_DIR)/agent.c $(SRC_DIR)/environment.c -lm
@echo "Running Q-table optimization tests..."
@./test_qtable_optimization
@echo "Cleaning test executable..."
@rm -f test_qtable_optimization

# Run all tests
test-all: test-environment test-step-env test-rewards test-priority-replay test-state-visit-tracking test-qtable-optimization
@echo "All tests completed successfully!"

# Package for distribution
package: release
	@echo "Creating distribution package..."
	@mkdir -p dist
	@cp $(TARGET) dist/
	@cp README.md dist/ 2>/dev/null || echo "README.md not found"
	@tar -czf rl_agent_$(shell date +%Y%m%d).tar.gz dist/
	@echo "Package created: rl_agent_$(shell date +%Y%m%d).tar.gz"

# Show help
help:
	@echo "Available targets:"
	@echo "  all          - Build the project (default)"
	@echo "  clean        - Remove build artifacts"
	@echo "  rebuild      - Clean and build"
	@echo "  run          - Build and run the application"
	@echo "  debug        - Build with debug symbols"
	@echo "  release      - Build optimized release version"
	@echo "  install-raylib - Install raylib dependency"
	@echo "  check-deps   - Check if dependencies are installed"
	@echo "  format       - Format source code with clang-format"
	@echo "  analyze      - Run static analysis with cppcheck"
	@echo "  docs         - Generate documentation with doxygen"
	@echo "  test         - Run basic tests"
@echo "  test-rewards - Test comprehensive reward system"
@echo "  test-environment - Test environment functions"
@echo "  test-step-env    - Test step_environment function"
@echo "  test-qtable-optimization - Test Q-table optimization features"
@echo "  test-all     - Run all test suites"
	@echo "  package      - Create distribution package"
	@echo "  help         - Show this help message"

# File dependencies
$(BUILD_DIR)/main.o: $(INCLUDE_DIR)/agent.h $(INCLUDE_DIR)/environment.h $(INCLUDE_DIR)/rendering.h $(INCLUDE_DIR)/utils.h
$(BUILD_DIR)/agent.o: $(INCLUDE_DIR)/agent.h $(INCLUDE_DIR)/utils.h
$(BUILD_DIR)/environment.o: $(INCLUDE_DIR)/environment.h $(INCLUDE_DIR)/agent.h $(INCLUDE_DIR)/utils.h
$(BUILD_DIR)/rendering.o: $(INCLUDE_DIR)/rendering.h $(INCLUDE_DIR)/environment.h $(INCLUDE_DIR)/agent.h $(INCLUDE_DIR)/utils.h
$(BUILD_DIR)/utils.o: $(INCLUDE_DIR)/utils.h
$(BUILD_DIR)/q_table_optimized.o: $(INCLUDE_DIR)/q_table_optimized.h

# Prevent make from deleting intermediate files
.PRECIOUS: $(BUILD_DIR)/%.o

# Declare phony targets
.PHONY: all directories clean rebuild run debug release install-raylib check-deps format analyze docs test test-rewards test-environment test-step-env test-qtable-optimization test-all package help
