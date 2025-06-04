# Compiler and flags
CXX := g++
CXXFLAGS := -std=c++11 -Wall -Wextra
LDFLAGS := -lX11 -lpthread

# Targets
TARGETS := server client

# Default target
all: $(TARGETS)

# Server build
server: server.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

# Client build
client: client.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

# Clean build artifacts
clean:
	rm -f $(TARGETS)

# Install dependencies (for Ubuntu/Debian)
install-deps:
	sudo apt-get install -y libx11-dev libxi-dev

.PHONY: all clean install-deps