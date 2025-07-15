#!/bin/bash

# Create directories
mkdir -p src include

# Create source files
touch src/main.c
touch src/environment.c
touch src/agent.c
touch src/rendering.c
touch src/utils.c

# Create header files
touch include/environment.h
touch include/agent.h
touch include/rendering.h

echo "Project structure created successfully."
