#
# 'make depend' uses makedepend to automatically generate dependencies
#               (dependencies are added to end of Makefile)
# 'make'        build executable file 'mycc'
# 'make clean'  removes all .o and executable files
#

# define the HIPCC compiler to use
CC = hipcc

# define any compile-time flags
CFLAGS = -Wall -O2 -g -std=c++11 $(shell hipconfig --cpp_config) # C flags

# define any directories containing header files other than /usr/include
#

ROCBLAS_DIR = .
COBLAT_DIR = /home/tim/Tensile/TensileLib

INCLUDES = -I$(ROCBLAS_DIR)/include  -I$(ROCBLAS_DIR)/src/include -I$(COBLAT_DIR)/include

# define library paths in addition to /usr/lib
#   if I wanted to include libraries not in /usr/lib I'd specify
#   their path using -Lpath, something like:
LFLAGS = -fpic # linking flags

ARCH = ar
ARCHFLAGS = -rcs
RANLIB = ranlib
# define the C source files

ROCBLAS_SRCS = $(ROCBLAS_DIR)/src/blas1/rocblas_dot.cpp \
       	$(ROCBLAS_DIR)/src/blas1/rocblas_scal.cpp \
       	$(ROCBLAS_DIR)/src/blas1/rocblas_copy.cpp \
       	$(ROCBLAS_DIR)/src/rocblas_auxiliary.cpp \
       	$(ROCBLAS_DIR)/src/blas2/rocblas_symv.cpp \
       	$(ROCBLAS_DIR)/src/blas3/rocblas_trtri.cpp \
       	$(ROCBLAS_DIR)/src/blas3/rocblas_trtri_batched.cpp \
       	$(ROCBLAS_DIR)/src/handle.cpp \
       	$(ROCBLAS_DIR)/src/status.cpp \


# define the C object files
#
# This uses Suffix Replacement within a macro:
#   $(name:string1=string2)
#         For each word in 'name' replace 'string1' with 'string2'
# Below we are replacing the suffix .c of all words in the macro SRCS
# with the .o suffix
#

ROCBLAS_OBJ = $(ROCBLAS_SRCS:.cpp=.o)


OBJS =  $(ROCBLAS_OBJ)


# define the executable file
TARGET_LIB = ./lib/librocblas.a

#
# The following part of the makefile is generic; it can be used to
# build any executable just by changing the definitions above and by
# deleting dependencies appended to the file from 'make depend'
#

.PHONY: depend clean

all:  $(TARGET_LIB)
#    @echo  Simple compiler named mycc has been compiled

$(TARGET_LIB): $(OBJS)
	$(ARCH) $(ARCHFLAGS) $(TARGET_LIB) $(OBJS)

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
	$(RM) $(OBJS) *~ $(TARGET_LIB)

depend: $(SRCS)
	makedepend $(INCLUDES) $^
