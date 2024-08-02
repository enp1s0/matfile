CXX=g++
CXXFLAGS=-std=c++17 -I../include

TARGETS=dense.test matrix_market.test

ifeq ($(TEST_OLD_FORMAT), 1)
	CXXFLAGS += -DMATFILE_USE_OLD_FORMAT
endif

all: $(TARGETS)

%.test:%.cpp
	$(CXX) $< -o $@ $(CXXFLAGS)
  
clean:
	rm -f $(TARGETS)
