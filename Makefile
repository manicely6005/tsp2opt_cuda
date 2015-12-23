#makefile for Traveling Salesman Problem

all:		tsp_cuda2opt

clean:
		rm -rf $(OBJDIR)/*o tsp_cuda2opt

CXX_PATH = g++
SRCDIR		:= src
OBJDIR		:= obj

ARCHS		:= -gencode arch=compute_20,code=sm_20 \
				-gencode arch=compute_30,code=sm_30 \
				-gencode arch=compute_32,code=sm_32 \
				-gencode arch=compute_35,code=sm_35 \
				-gencode arch=compute_50,code=sm_50 \
				-gencode arch=compute_52,code=sm_52
				
CPPFLAGS	:= -O3 -g -fPIC -Wall -Wextra -std=c++11
NVCCFLAGS  	:= -O3 -G -std=c++11 -w --use_fast_math -res-usage -lineinfo -Xptxas -v
CUDA_LINK_FLAGS := -lcuda -lcudart 

OBJECTS :=\
	$(OBJDIR)/main.o\
	$(OBJDIR)/algorithms.o\
	$(OBJDIR)/edge_weight.o\
	$(OBJDIR)/opt_kernel.o\
	$(OBJDIR)/wrapper.o

OUT_FILES = $(wildcard $(OBJDIR)/*.o)

OS 		:= $(shell uname)
CPUARCH := $(shell uname -p)
HOST 	:= $(shell hostname | awk -F. '{print $$1}')

ifeq ($(OS),Linux)
#LINUX
  ifeq ($(CPUARCH), x86_64)
    CUDA_LIBRARY_DIR = /usr/local/cuda/lib64
    ARCHFLAG := -m64
  else
    CUDA_LIBRARY_DIR = /usr/local/cuda/lib
    ARCHFLAG := -m32
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
CPPFLAGS	+= $(ARCHFLAG)
NVCCFLAGS	+= $(ARCHFLAG)

info:
	@echo
	@echo "HOST =" $(HOST)
	@echo "DETECTED OS =" $(OS)
	@echo "DETECTED ARCH =" $(CPUARCH)
	@echo "DETECTED" $(NPROCS) "CPUs"
	@echo

tsp_cuda2opt: info $(OBJECTS)
	$(CXX_PATH) -o tsp_cuda2opt $(OUT_FILES) $(LINKFLAGS) $(CPPFLAGS)

$(OBJDIR)/main.o: $(SRCDIR)/main.cpp $(SRCDIR)/algorithms.h
	$(CXX_PATH) $(CPPFLAGS) -o $(OBJDIR)/main.o -c $(SRCDIR)/main.cpp

$(OBJDIR)/algorithms.o: $(SRCDIR)/algorithms.cpp $(SRCDIR)/algorithms.h $(SRCDIR)/wrapper.cuh
	$(CXX_PATH) $(CPPFLAGS) -o $(OBJDIR)/algorithms.o -c $(SRCDIR)/algorithms.cpp

$(OBJDIR)/edge_weight.o: $(SRCDIR)/edge_weight.cpp $(SRCDIR)/edge_weight.h $(SRCDIR)/algorithms.h
	$(CXX_PATH) $(CPPFLAGS) -o $(OBJDIR)/edge_weight.o -c $(SRCDIR)/edge_weight.cpp
	
$(OBJDIR)/opt_kernel.o: $(SRCDIR)/opt_kernel.cu $(SRCDIR)/opt_kernel.cuh $(SRCDIR)/algorithms.h
	$(NVCC_PATH) $(NVCCFLAGS) $(ARCHS) -L$(CUDA_LIBRARY_DIR) -L$(CUDA_BIN_DIR) -I$(CUDA_INCLUDE_DIR) -o $(OBJDIR)/opt_kernel.o -c $(SRCDIR)/opt_kernel.cu
	
$(OBJDIR)/wrapper.o: $(SRCDIR)/wrapper.cu $(SRCDIR)/wrapper.cuh $(SRCDIR)/opt_kernel.cuh
	$(NVCC_PATH) $(NVCCFLAGS) $(ARCHS) -L$(CUDA_LIBRARY_DIR) -L$(CUDA_BIN_DIR) -I$(CUDA_INCLUDE_DIR) -o $(OBJDIR)/wrapper.o -c $(SRCDIR)/wrapper.cu
