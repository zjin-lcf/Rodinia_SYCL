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

# Enable checksum and output file
ifeq ($(DEBUG),yes)
  CFLAGS += -g  
  TEST_ON = -DTEST_CHECKSUM 
  OUTPUT  = -DOUTPUT
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

ifeq ($(DEVICE),gpu)
  CFLAGS +=-DUSE_GPU
endif


heartwall:	./main.o \
	    ./kernel/kernel.o \
			./util/avi/avilib.o \
			./util/avi/avimod.o \
			./util/file/file.o \
			./util/timer/timer.o 
	$(CC)	$(CFLAGS) ./main.o \
		  ./kernel/kernel.o \
			./util/avi/avilib.o \
			./util/avi/avimod.o \
			./util/file/file.o \
			./util/timer/timer.o \
			$(LDFLAGS) -o heartwall

# ======================================================================================================================================================150
#	MAIN FUNCTION
# ======================================================================================================================================================150

./main.o:	./main.h \
			./main.c
	$(CC)	$(OUTPUT) $(KERNEL_DIM) ./main.c \
		  -I./util/timer/ \
		  -I./util/file/ \
			-c \
			-o ./main.o 

# ======================================================================================================================================================150
#	KERNELS
# ======================================================================================================================================================150

./kernel/kernel.o:./kernel/kernel.sycl ./kernel/kernel.cpp
	$(CC) $(CFLAGS) $(TEST_ON) $(KERNEL_DIM)	./kernel/kernel.cpp \
			-c \
			-o ./kernel/kernel.o 

# ======================================================================================================================================================150
#	UTILITIES
# ======================================================================================================================================================150

./util/avi/avilib.o:./util/avi/avilib.h \
					./util/avi/avilib.c
	$(CC)	$(CFLAGS) ./util/avi/avilib.c \
		  -I./util/avi \
			-c \
			-o ./util/avi/avilib.o 

./util/avi/avimod.o:./util/avi/avilib.h \
					./util/avi/avimod.c
	$(CC)	$(CFLAGS) ./util/avi/avimod.c \
		  -I./util/avi \
			-c \
			-o ./util/avi/avimod.o 

./util/file/file.o:	./util/file/file.h \
					./util/file/file.c
	$(CC)	$(CFLAGS) ./util/file/file.c \
		  -I./util/file \
		  -I./util/timer \
			-c \
			-o ./util/file/file.o 

./util/timer/timer.o:	./util/timer/timer.h \
						./util/timer/timer.c
	$(CC)	$(CFLAGS) ./util/timer/timer.c \
		  -I./util/timer \
			-c \
			-o ./util/timer/timer.o 

run:
	./heartwall 20

clean:
	rm	-rf *.o heartwall \
		./kernel/*.o \
		./util/avi/*.o \
		./util/file/*.o \
		./util/timer/*.o 
