#makefile for Traveling Salesman Problem

all:		tsp_cuda2opt

clean:
		rm -rf $(OBJDIR)/*o tsp_cuda2opt
		
debug:		CPPFLAGS += -g
debug:		NVCCFLAGS += -G -lineinfo
debug:		tsp_cuda2opt

CXX	 	?= g++
SRCDIR		:= src
OBJDIR		:= obj

ARCHS		:= -gencode arch=compute_20,code=sm_20 \
		-gencode arch=compute_30,code=sm_30 \
		-gencode arch=compute_32,code=sm_32 \
		-gencode arch=compute_35,code=sm_35 \
		-gencode arch=compute_50,code=sm_50 \
		-gencode arch=compute_52,code=sm_52
	
CXXFLAGS	:= -O3 -Wall -Wextra -std=c++11
NVCCFLAGS  	:= -O3 -std=c++11 -w --use_fast_math -Xptxas -v
CUDA_LINK_FLAGS := -lcuda -lcudart 
OMPI		:= $(shell mpicc --showme:compile)

OBJECTS 	:=\
		$(OBJDIR)/opt_kernel.o\
		$(OBJDIR)/wrapper.o\
		$(OBJDIR)/edge_weight.o\
		$(OBJDIR)/algorithms.o\
		$(OBJDIR)/main.o

OS 		:= $(shell uname)
CPUARCH 	:= $(shell uname -p)
HOST 		:= $(shell hostname | awk -F. '{print $$1}')

ifeq ($(OS),Linux)
#LINUX
  ifeq ($(CPUARCH), x86_64)
    CUDA_LIBRARY_DIR = /usr/local/cuda/lib64
    ARCHFLAG := -m64
  else ifeq ($(CPUARCH), i386)
    CUDA_LIBRARY_DIR = /usr/local/cuda/lib
    ARCHFLAG := -m32
  else
    CUDA_LIBRARY_DIR = /usr/local/cuda/lib
    ARCHFLAG := -D ARM	# Define ARM. For initialization of mapped memory
  endif # $(CPUARCH)
  CUDA_INCLUDE_DIR = /usr/local/cuda/include
  NVCC_PATH = /usr/local/cuda/bin/nvcc
  NPROCS := $(shell grep -c ^processor /proc/cpuinfo)
  
else ifeq ($(OS),Darwin)
#MAC OS - mine at least
  CUDA_LIBRARY_DIR = /usr/local/cuda/lib
  CUDA_INCLUDE_DIR = /Developer/NVIDIA/CUDA-7.5/include
  NVCC_PATH = /Developer/NVIDIA/CUDA-7.5/bin/nvcc
  ARCHFLAG := -m64 
  NPROCS := $(shell sysctl hw.ncpu | awk '{print $$2}')
endif

LINKFLAGS	:= -L$(CUDA_LIBRARY_DIR) -I$(CUDA_INCLUDE_DIR) $(CUDA_LINK_FLAGS)
CXXFLAGS	+= $(ARCHFLAG)
NVCCFLAGS	+= $(ARCHFLAG)

info:
	@echo
	@echo "HOST =" $(HOST)
	@echo "DETECTED OS =" $(OS)
	@echo "DETECTED ARCH =" $(CPUARCH)
	@echo "DETECTED" $(NPROCS) "CPUs"
	@echo

tsp_cuda2opt: info $(OBJECTS)
	$(CXX) $(OBJECTS) -o tsp_cuda2opt $(LINKFLAGS) $(CXXFLAGS) $(OMPI)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(OMPI) -o $@ -c $^
	
$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	$(NVCC_PATH) $(NVCCFLAGS) $(ARCHS) $(LINKFLAGS) -o $@ -c $^
