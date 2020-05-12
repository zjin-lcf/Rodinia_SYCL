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
  CFLAGS += -g  
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

ifeq ($(DEVICE),gpu)
  CFLAGS +=-DUSE_GPU
endif



EXE = hotspot

$(EXE): hotspot.cpp hotspot.h 
	$(CC) $(CFLAGS) -o $(EXE) hotspot.cpp $(LDFLAGS)

run: $(EXE)
	./$(EXE) 64 2 2 ../../data/hotspot/temp_64 ../../data/hotspot/power_64 output_64.out
	./$(EXE) 512 2 2 ../../data/hotspot/temp_512 ../../data/hotspot/power_512 output_512.out

clean:
	rm -f $(EXE)
