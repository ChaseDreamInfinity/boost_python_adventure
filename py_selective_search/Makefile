PROJECT := py_selective_search

DYNAMIC_NAME := $(PROJECT).so

ANACONDA_HOME := $(HOME)/anaconda3
PYTHON_INCLUDE := $(ANACONDA_HOME)/include/python3.5m
PYTHON_LIBRARIES := boost_python3
WARNINGS := -Wall -Wno-sign-compare -Wno-unused-function -Wno-unused-variable

OPENCV_PKG_CONFIG := $(shell pkg-config opencv --libs)

LDFLAGS += $(OPENCV_PKG_CONFIG)

INCLUDE_DIRS += $(PYTHON_INCLUDE) $(wildcar *.h)

CXX=/usr/bin/g++
COMMON_FLAGS += -DNDEBUG -O2

PYTHON_LDFLAGS := $(LDFLAGS) $(foreach library,$(PYTHON_LIBRARIES),-l$(library))
COMMON_FLAGS += $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir))
CXXFLAGS += -pthread -fPIC -std=c++11 $(COMMON_FLAGS) $(WARNINGS)

CXX_SRCS := $(shell find -name "*.cpp")
CXX_OBJS := $(CXX_SRCS:.cpp=.o) 

########################
# Define build targets
#######################
.PHONY: all lib clean

all: lib

lib: $(DYNAMIC_NAME)

$(DYNAMIC_NAME): $(CXX_OBJS) 
	$(CXX) -shared -o $@ $(CXX_OBJS) $(PYTHON_LDFLAGS)

$(CXX_OBJS): $(CXX_SRCS)
	$(CXX) -o $@ $(CXXFLAGS) -c $<

clean:
	@- $(RM) -rf $(CXX_OBJS) $(DYNAMIC_NAME)
