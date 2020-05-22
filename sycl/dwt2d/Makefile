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
  CFLAGS += -g -DOUTPUT -DDEBUG
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

ifeq ($(DEVICE),gpu)
  CFLAGS +=-DUSE_GPU
endif


dwt2d: main.cpp
	$(CC) $(CFLAGS) -o dwt2d \
		main.cpp \
		$(OUTPUT) \
		$(LDFLAGS)
		
run: dwt2d
	./dwt2d 192.bmp -d 192x192 -f -5 -l 3
	./dwt2d rgb.bmp -d 1024x1024 -f -5 -l 3

clean:	
	rm -rf *.o dwt2d
	rm -rf *.bmp.dwt.*

