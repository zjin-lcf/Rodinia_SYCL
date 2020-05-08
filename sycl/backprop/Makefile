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
  CFLAGS += -g  -DOUTPUT
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

ifeq ($(DEVICE),gpu)
  CFLAGS +=-DUSE_GPU
endif


backprop: backprop.o imagenet.o facetrain.o
	$(CC) $(CFLAGS) backprop_sycl.cpp backprop.o imagenet.o facetrain.o -o backprop $(LDFLAGS)

%.o: %.[ch]
	$(CC) $(CFLAGS) $< -c
	
backprop.o: backprop.c 
	$(CC) $(CFLAGS) backprop.c -c
	
facetrain.o: facetrain.c
	$(CC) $(CFLAGS) facetrain.c -c

imagenet.o: imagenet.c
	$(CC) $(CFLAGS) imagenet.c -c

run: backprop
	./backprop 65536

clean:
	rm -f *.o *~ backprop *.linkinfo
