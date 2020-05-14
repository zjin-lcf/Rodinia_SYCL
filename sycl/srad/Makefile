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

./srad:	./main.o \
			./kernel/kernel_gpu_opencl_wrapper.o \
			./util/graphics/graphics.o \
			./util/graphics/resize.o \
			./util/timer/timer.o
	$(CC)	./main.o \
			./kernel/kernel_gpu_opencl_wrapper.o \
			./util/graphics/graphics.o \
			./util/graphics/resize.o \
			./util/timer/timer.o \
			-lm \
			$(LDFLAGS) -o srad

# ======================================================================================================================================================150
#	MAIN FUNCTION
# ======================================================================================================================================================150

./main.o:	./main.h \
			./main.c
	$(CC)	$(KERNEL_DIM) ./main.c \
			-c \
			-o ./main.o \
			-O3

# ======================================================================================================================================================150
#	KERNELS
# ======================================================================================================================================================150

./kernel/kernel_gpu_opencl_wrapper.o:	./kernel/kernel_gpu_opencl_wrapper.h \
										./kernel/kernel_gpu_opencl_wrapper.c
	$(CC) $(CFLAGS)	$(KERNEL_DIM) ./kernel/kernel_gpu_opencl_wrapper.c \
			-c \
			-o ./kernel/kernel_gpu_opencl_wrapper.o 

# ======================================================================================================================================================150
#	UTILITIES
# ======================================================================================================================================================150

./util/graphics/graphics.o:	./util/graphics/graphics.h \
							./util/graphics/graphics.c
	$(CC)	./util/graphics/graphics.c \
			-c \
			-o ./util/graphics/graphics.o \
			-O3

./util/graphics/resize.o:	./util/graphics/resize.h \
							./util/graphics/resize.c
	$(CC)	./util/graphics/resize.c \
			-c \
			-o ./util/graphics/resize.o \
			-O3

./util/timer/timer.o:	./util/timer/timer.h \
						./util/timer/timer.c
	$(CC)	./util/timer/timer.c \
			-c \
			-o ./util/timer/timer.o \
			-O3

run: srad
	./srad 100 0.5 502 458

clean:
	rm	./*.o srad\
		./kernel/*.o \
		./util/graphics/*.o \
		./util/timer/*.o
