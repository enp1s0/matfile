CXX=g++
CXXFLAGS=-std=c++14 -I../include

TARGETS=dense.test

all: $(TARGETS)

%.test:%.cpp
	$(CXX) $< -o $@ $(CXXFLAGS)
  
clean:
	rm -f $(TARGET)
