VENDOR      = intel
OPTIMIZE    = yes
DEBUG       = no
DEVICE      = gpu



# Standard flags (profiling is enabled to measure device offloading time)
CFLAGS := -std=c++11 -Wall -DPROFILING

# Linker flags
LDFLAGS = -lm

# Codeplay Compiler
ifneq (,$(findstring codeplay,$(VENDOR)))
  CC = compute++
  CFLAGS += -no-serial-memop -sycl -sycl-driver
  LDFLAGS += -lComputeCpp
endif

# Intel DPC++ Compiler
ifneq (,$(findstring intel,$(VENDOR)))
  CC = dpcpp
  CFLAGS += 
  LDFLAGS += 
endif

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g 
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

ifeq ($(DEVICE),gpu)
  CFLAGS +=-DUSE_GPU
endif

SRC = bfs.cpp timer.cc

EXE = bfs

$(EXE):$(SRC)
	$(CC) $(CFLAGS)   $(SRC) -o $(EXE)  $(LDFLAGS) 

run: $(EXE)
	./$(EXE) ../../data/bfs/graph1MW_6.txt
	#./$(EXE) ../../data/bfs/graph4096.txt
	#./$(EXE) ../../data/bfs/graph65536.txt


clean: 
	rm -f $(EXE) $(EXE).linkinfo result*
