# Makefile for Problem 1 in project
# Compiles C++ source files with MPI and OpenCV dependencies

# Compiler and flags
CXX = mpic++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2
LDFLAGS = `pkg-config --libs opencv4`

# Include directories
INCLUDES = `pkg-config --cflags opencv4`

# Source files
SOURCES = main.cpp MNIST.cpp Kmeans.cpp BaseKmeans.cpp Clusters.cpp ImageVectors.cpp

# Object files
OBJECTS = $(SOURCES:.cpp=.o)

# Executable name
EXEC = kmeans_federated

# Default target
all: $(EXEC)

# Link object files
$(EXEC): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $@ $(LDFLAGS)

# Compile source files to object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean up
clean:
	rm -f $(OBJECTS) $(EXEC)

# Phony targets
.PHONY: all clean