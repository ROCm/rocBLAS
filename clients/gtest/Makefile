#
# 'make depend' uses makedepend to automatically generate dependencies
#               (dependencies are added to end of Makefile)
# 'make'        build executable file 'mycc'
# 'make clean'  removes all .o and executable files
#

# define the C compiler to use
CC = g++
HCC = hipcc

# define any compile-time flags
CFLAGS = -Wall -g -std=c++11  $(shell hipconfig --cpp_config) -DGTEST_USE_OWN_TR1_TUPLE=1

#clang lib
#-stdlib=libc++
# define any directories containing header files other than /usr/include
#

ROCBLAS_DIR = /home/tim/aBLAS/library
COBLAT_DIR = /home/tim/Cobalt/CobaltLib

INCLUDES = -I$(ROCBLAS_DIR)/include  -I$(ROCBLAS_DIR)/src/include -I../include -I$(COBLAT_DIR)/include

# define library paths in addition to /usr/lib
#   if I wanted to include libraries not in /usr/lib I'd specify
#   their path using -Lpath, something like:
LFLAGS =

# define any libraries to link into executable:
#   if I want to link in libraries (libx.so or libx.a) I use the -llibname
#   option, something like (this will link in libmylib.so and libm.so:

#system gtest lib
GTEST_LIBS := -lgtest -lgtest_main -lpthread #-L/opt/rocm/hcc/compiler/lib

LIBS = -lblas -llapack $(GTEST_LIBS)


# define the C source files

ROCBLAS_SRCS =  \
       	$(ROCBLAS_DIR)/src/handle.cpp \
       	$(ROCBLAS_DIR)/src/status.cpp \
       	$(ROCBLAS_DIR)/src/rocblas_auxiliary.cpp \


#		$(ROCBLAS_DIR)/src/blas1/rocblas_dot.cpp \
       	$(ROCBLAS_DIR)/src/blas1/rocblas_scal.cpp \
       	$(ROCBLAS_DIR)/src/blas1/rocblas_copy.cpp \
       	$(ROCBLAS_DIR)/src/blas2/rocblas_symv.cpp \
       	$(ROCBLAS_DIR)/src/blas3/rocblas_trtri.cpp \
       	$(ROCBLAS_DIR)/src/blas3/rocblas_trtri_batched.cpp \

COMMON_SRCS = ../common/cblas_interface.cpp \
              ../common/norm.cpp \
              ../common/unit.cpp \
              ../common/utility.cpp \
              ../common/flops.cpp \

GTEST_SRCS = ./rocblas_gtest_main.cpp \
			blas1_gtest.cpp \
			gemm_gtest.cpp \
			trsm_gtest.cpp \

#			gemm_gtest.cpp \

# define the C object files
#
# This uses Suffix Replacement within a macro:
#   $(name:string1=string2)
#         For each word in 'name' replace 'string1' with 'string2'
# Below we are replacing the suffix .c of all words in the macro SRCS
# with the .o suffix
#

ROCBLAS_OBJ = $(ROCBLAS_SRCS:.cpp=.o)
COMMON_OBJ = $(COMMON_SRCS:.cpp=.o)
GTEST_OBJ = $(GTEST_SRCS:.cpp=.o)

OBJS =  $(ROCBLAS_OBJ) $(COMMON_OBJ) $(GTEST_OBJ)

# define the executable file
MAIN = rocblas_gtest

#
# The following part of the makefile is generic; it can be used to
# build any executable just by changing the definitions above and by
# deleting dependencies appended to the file from 'make depend'
#

.PHONY: depend clean

all:  $(MAIN)
#    @echo  Simple compiler named mycc has been compiled

$(MAIN): $(OBJS)
	$(HCC) $(CFLAGS) $(INCLUDES) -o $(MAIN) $(OBJS) $(LFLAGS) $(LIBS)

# this is a suffix replacement rule for building .o's from .c's
# it uses automatic variables $<: the name of the prerequisite of
# the rule(a .c file) and $@: the name of the target of the rule (a .o file)
# (see the gnu make manual section about automatic variables)

.cpp.o:
$(ROCBLAS_OBJ): $(ROCBLAS_SRCS)
	$(HCC) $(CFLAGS) $(INCLUDES) -c $<  -o $@
$(COMMON_OBJ): $(COMMON_SRCS)
	$(CC) $(CFLAGS) $(INCLUDES) -c $<  -o $@
$(GTEST_OBJ): $(GTEST_SRCS)
	$(CC) $(CFLAGS) $(INCLUDES) -c $<  -o $@

# %.o: %.cpp
# 	$(CC) $(CFLAGS) $(INCLUDES) -c $<  -o $@

# .cpp.o:
# 	$(CC) $(CFLAGS) $(INCLUDES) -c $<  -o $@

clean:
	$(RM) $(OBJS) *~ $(MAIN)

depend: $(GTEST_SRCS) $(ROCBLAS_SRCS) $(COMMON_SRCS)
	makedepend $(INCLUDES) $^
