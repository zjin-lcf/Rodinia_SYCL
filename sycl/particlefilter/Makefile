VENDOR      = intel
OPTIMIZE    = yes
DEBUG       = no
DEVICE      = gpu


# turning the work-group size
BLOCK_SIZE=512

# Standard flags
CFLAGS := -std=c++11 -Wall -DBLOCK_SIZE=$(BLOCK_SIZE)

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

# Enable checksum and output file
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


ALL: SYCL_particlefilter_single

SYCL_particlefilter_single: ex_particle_SYCL_single_seq.o
	$(CC) $(CFLAGS) ex_particle_SYCL_single_seq.o $(LDFLAGS)  -o SYCL_particlefilter_single 

%.o: %.[ch]
	$(CC) $(CFLAGS) $< -c

ex_particle_SYCL_single_seq.o: ex_particle_SYCL_single_seq.cpp
	$(CC) $(CFLAGS) ex_particle_SYCL_single_seq.cpp -c

run: SYCL_particlefilter_single
	./SYCL_particlefilter_single -x 128 -y 128 -z 10 -np 400000

clean:
	rm -f *.o *~ SYCL_particlefilter_single 
