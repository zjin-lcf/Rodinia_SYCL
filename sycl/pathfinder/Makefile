VENDOR      = intel
OPTIMIZE    = yes
DEBUG       = no
DEVICE      = gpu



# Standard flags
CFLAGS := -std=c++11 -Wall

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
  CFLAGS += -g -DBENCH_PRINT
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

ifeq ($(DEVICE),gpu)
  CFLAGS +=-DUSE_GPU
endif


# link objects(binaries) together
pathfinder: main.o 
	$(CC) main.o $(LDFLAGS) -o pathfinder

# compile main function file into object (binary)
main.o: main.cpp  kernel.sycl
	$(CC) $(CFLAGS) main.cpp -c -o main.o

run: pathfinder
	./pathfinder 100000 100 20 > result.txt

clean:
	@rm -f pathfinder *.o *.out *~ *\#
