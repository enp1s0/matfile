CXX=g++
CXXFLAGS=-std=c++17 -I../include -fopenmp

TARGET=matfile-comp

$(TARGET):src/comp.cpp
	$(CXX) $< -o $@ $(CXXFLAGS)
  
clean:
	rm -f $(TARGET)
