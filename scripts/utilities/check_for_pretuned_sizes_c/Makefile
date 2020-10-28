# Makefile assumes rocBLAS is installed in /opt/rocm/rocblas

ROCM_INSTALL_DIR=/opt/rocm
ROCBLAS_INSTALL_DIR=$(ROCM_INSTALL_DIR)
ROCBLAS_INCLUDE=$(ROCBLAS_INSTALL_DIR)/include
ROCBLAS_LIB_PATH=$(ROCBLAS_INSTALL_DIR)/lib
HIP_INCLUDE=$(ROCM_INSTALL_DIR)/hip/include
LDFLAGS=-L$(ROCBLAS_LIB_PATH) -lrocblas -lboost_program_options
LD=$(ROCM_INSTALL_DIR)/bin/hipcc
CFLAGS=-I$(ROCBLAS_INCLUDE) -I$(HIP_INCLUDE) -std=c++14 -DROCM_USE_FLOAT16
CPP=$(ROCM_INSTALL_DIR)/bin/hipcc
OBJ=check-for-pretuned-sizes.o
EXE=check-for-pretuned-sizes

%.o: %.cpp
	$(CPP) -c -o $@ $< $(CFLAGS)

$(EXE) : $(OBJ)
	$(LD) $(OBJ) $(LDFLAGS) -o $@

clean:
	rm -f $(EXE) $(OBJ)
