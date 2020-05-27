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


# link objects together
lavaMD:	main.o \
	./kernel/kernel_gpu_opencl_wrapper.o \
	./util/num/num.o \
	./util/timer/timer.o 
	$(CC)	main.o \
		./kernel/kernel_gpu_opencl_wrapper.o \
		./util/num/num.o \
		./util/timer/timer.o \
		-lm \
		$(LDFLAGS) \
		-o lavaMD

# compile function files into objects
main.o:	main.h main.c 
	$(CC)	$(OUTPUT) main.c \
		-c \
		-o main.o \
		-O3

./kernel/kernel_gpu_opencl_wrapper.o:	./kernel/kernel_gpu_opencl_wrapper.h \
	./kernel/kernel_gpu_opencl_wrapper.c \
	./kernel/kernel.sycl 
	$(CC) $(CFLAGS)	$(KERNEL_DIM) ./kernel/kernel_gpu_opencl_wrapper.c \
		-c \
		-o ./kernel/kernel_gpu_opencl_wrapper.o \
		-O3

./util/num/num.o:	./util/num/num.h ./util/num/num.c
	$(CC) ./util/num/num.c \
		-c \
		-o ./util/num/num.o \
		-O3

./util/timer/timer.o:	./util/timer/timer.h ./util/timer/timer.c
	$(CC) ./util/timer/timer.c \
		-c \
		-o ./util/timer/timer.o \
		-O3

run: lavaMD
	./lavaMD -boxes1d 10

# delete all object and executable files
clean:
	rm *.o lavaMD \
		./kernel/*.o \
		./util/num/*.o \
		./util/timer/*.o \
