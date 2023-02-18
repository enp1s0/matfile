CXX=g++
CXXFLAGS=-std=c++17 -I../include -fopenmp

all: matfile-comp matfile-print matfile-info

matfile-%:src/%.cpp
	$(CXX) $< -o $@ $(CXXFLAGS)

matfile-info:src/info.cpp
	git submodule update --init
	$(CXX) $< -o $@ $(CXXFLAGS) -I./src/fphistogram/include
  
clean:
	rm -f $(TARGET)
