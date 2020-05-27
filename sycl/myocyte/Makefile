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

# ========================================================================================================================================================================================================200
#	EXECUTABLES (LINK OBJECTS TOGETHER INTO BINARY)
# ========================================================================================================================================================================================================200

./myocyte.out:	./main.o \
	./kernel/kernel_gpu_opencl_wrapper.o \
	./util/file/file.o \
	./util/num/num.o \
	./util/timer/timer.o 
	$(CC)	./main.o \
		./kernel/kernel_gpu_opencl_wrapper.o \
		./util/file/file.o \
		./util/num/num.o \
		./util/timer/timer.o \
		$(LDFLAGS) \
		-o myocyte.out

# ======================================================================================================================================================150
#	MAIN FUNCTION
# ======================================================================================================================================================150

./main.o:	./common.h \
	./main.h \
	./main.c
	$(CC) $(CFLAGS)	./main.c \
		-c \
		-o ./main.o 

# ======================================================================================================================================================150
#	KERNELS
# ======================================================================================================================================================150

./kernel/kernel_gpu_opencl_wrapper.o:	./common.h \
	./kernel/kernel_gpu_opencl_wrapper.h \
	./kernel/kernel_gpu_opencl_wrapper.c \
	./kernel/solver.c \
	./kernel/embedded_fehlberg_7_8.c \
	./kernel/master.c \
	./kernel/kernel_cam.sycl \
	./kernel/kernel_ecc.sycl
	$(CC) $(CFLAGS)	./kernel/kernel_gpu_opencl_wrapper.c \
		-c \
		-o ./kernel/kernel_gpu_opencl_wrapper.o 

# ======================================================================================================================================================150
#	UTILITIES
# ======================================================================================================================================================150

./util/file/file.o:	./common.h \
	./util/file/file.h \
	./util/file/file.c
	$(CC)	./util/file/file.c \
		-c \
		-o ./util/file/file.o \
		-O3

./util/num/num.o:	./common.h \
	./util/num/num.h \
	./util/num/num.c
	$(CC)	./util/num/num.c \
		-c \
		-o ./util/num/num.o \
		-O3

./util/timer/timer.o:	./common.h \
	./util/timer/timer.h \
	./util/timer/timer.c
	$(CC)	./util/timer/timer.c \
		-c \
		-o ./util/timer/timer.o \
		-O3


run: myocyte.out
	./myocyte.out -time 100

clean:
	rm	*.o *.out \
		./kernel/*.o \
		./util/file/*.o \
		./util/timer/*.o \
		./util/num/*.o \
		./output.txt
