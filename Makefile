# Makefile for RL Agent Environment with Raylib
# Author: Auto-generated for C Raylib RL Simulation

# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -O2 -g
INCLUDES = -Iinclude
LIBS = -lraylib -lm

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
	@echo "  package      - Create distribution package"
	@echo "  help         - Show this help message"

# File dependencies
$(BUILD_DIR)/main.o: $(INCLUDE_DIR)/agent.h $(INCLUDE_DIR)/environment.h $(INCLUDE_DIR)/rendering.h $(INCLUDE_DIR)/utils.h
$(BUILD_DIR)/agent.o: $(INCLUDE_DIR)/agent.h $(INCLUDE_DIR)/utils.h
$(BUILD_DIR)/environment.o: $(INCLUDE_DIR)/environment.h $(INCLUDE_DIR)/agent.h $(INCLUDE_DIR)/utils.h
$(BUILD_DIR)/rendering.o: $(INCLUDE_DIR)/rendering.h $(INCLUDE_DIR)/environment.h $(INCLUDE_DIR)/agent.h $(INCLUDE_DIR)/utils.h
$(BUILD_DIR)/utils.o: $(INCLUDE_DIR)/utils.h

# Prevent make from deleting intermediate files
.PRECIOUS: $(BUILD_DIR)/%.o

# Declare phony targets
.PHONY: all directories clean rebuild run debug release install-raylib check-deps format analyze docs test package help
