#
# 'make depend' uses makedepend to automatically generate dependencies
#               (dependencies are added to end of Makefile)
# 'make'        build executable file 'mycc'
# 'make clean'  removes all .o and executable files
#

# define the C compiler to use
CC = hipcc -fopenmp

# define any compile-time flags
CFLAGS = -Wall -g -std=c++11 $(shell hipconfig --cpp_config)

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
LIBS = -lblas -llapack #-L$(ROCBLAS_DIR)/lib/ -lrocblas until rocm1.1 link with .a does not work

# define the C source files

ROCBLAS_SRCS = $(ROCBLAS_DIR)/src/blas1/rocblas_dot.cpp \
       	$(ROCBLAS_DIR)/src/blas1/rocblas_scal.cpp \
       	$(ROCBLAS_DIR)/src/blas1/rocblas_copy.cpp \
       	$(ROCBLAS_DIR)/src/blas1/rocblas_axpy.cpp \
       	$(ROCBLAS_DIR)/src/blas1/rocblas_asum.cpp \
       	$(ROCBLAS_DIR)/src/blas1/rocblas_nrm2.cpp \
       	$(ROCBLAS_DIR)/src/blas1/fetch_template.cpp \
       	$(ROCBLAS_DIR)/src/rocblas_auxiliary.cpp \
       	$(ROCBLAS_DIR)/src/blas2/rocblas_symv.cpp \
       	$(ROCBLAS_DIR)/src/blas2/rocblas_gemv.cpp \
       	$(ROCBLAS_DIR)/src/blas3/rocblas_trtri.cpp \
       	$(ROCBLAS_DIR)/src/blas3/rocblas_trtri_batched.cpp \
       	$(ROCBLAS_DIR)/src/handle.cpp \
       	$(ROCBLAS_DIR)/src/status.cpp \

COMMON_SRCS = ../common/cblas_interface.cpp \
              ../common/norm.cpp \
              ../common/unit.cpp \
              ../common/utility.cpp \
              ../common/flops.cpp \

OPENMP_SRCS = ./example_openmp.cpp
SCAL_SRCS = ./example_sscal.cpp
CLIENT_SRCS = ./client.cpp

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
OPENMP_OBJ = $(OPENMP_SRCS:.cpp=.o)
SCAL_OBJ = $(SCAL_SRCS:.cpp=.o)
CLIENT_OBJ = $(CLIENT_SRCS:.cpp=.o)

OPENMP_OBJS =  $(COMMON_OBJ) $(OPENMP_OBJ) $(ROCBLAS_OBJ)
SCAL_OBJS =  $(COMMON_OBJ) $(SCAL_OBJ) $(ROCBLAS_OBJ)
CLIENT_OBJS = $(COMMON_OBJ) $(CLIENT_OBJ) $(ROCBLAS_OBJ)


# define the executable file
openmp = example_openmp
scal = example_sscal
client = client
#
# The following part of the makefile is generic; it can be used to
# build any executable just by changing the definitions above and by
# deleting dependencies appended to the file from 'make depend'
#

.PHONY: depend clean

all:  $(openmp) $(scal) $(client)
#    @echo  Simple compiler named mycc has been compiled

$(openmp): $(OPENMP_OBJS)
	$(CC) $(CFLAGS) $(INCLUDES) -o $(openmp) $(OPENMP_OBJS) $(LFLAGS) $(LIBS)

$(scal): $(SCAL_OBJS)
	$(CC) $(CFLAGS) $(INCLUDES) -o $(scal) $(SCAL_OBJS) $(LFLAGS) $(LIBS)

$(client): $(CLIENT_OBJS)
	$(CC) $(CFLAGS) $(INCLUDES) -o $(client) $(CLIENT_OBJS) $(LFLAGS) $(LIBS)

# this is a suffix replacement rule for building .o's from .c's
# it uses automatic variables $<: the name of the prerequisite of
# the rule(a .c file) and $@: the name of the target of the rule (a .o file)
# (see the gnu make manual section about automatic variables)

#.cpp.o:
%.o: %.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -c $<  -o $@

#.cpp.o:
#	$(CC) $(CFLAGS) $(INCLUDES) -c $<  -o $@

clean:
	$(RM) $(OPENMP_OBJS) $(SCAL_OBJS) $(CLIENT_OBJS) *~ $(openmp) $(scal) $(client)

depend: $(SRCS)
	makedepend $(INCLUDES) $^
