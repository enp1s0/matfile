CXX=g++
CXXFLAGS=-std=c++17 -I../include -fopenmp

all: matfile-comp

matfile-%:src/%.cpp
	$(CXX) $< -o $@ $(CXXFLAGS)
  
clean:
	rm -f $(TARGET)
