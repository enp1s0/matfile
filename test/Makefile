CXX=g++
CXXFLAGS=-std=c++17 -I../include

TARGETS=dense.test matrix_market.test

all: $(TARGETS)

%.test:%.cpp
	$(CXX) $< -o $@ $(CXXFLAGS)
  
clean:
	rm -f $(TARGETS)
