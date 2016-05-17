
void initialize_tensor_for_gemm(
    CobaltTensor & tensor,
    CobaltDataType data_type,
    int stride0,      // stride from one row to another
    int size0,        // num columns
    int stride1,      // stride from one column to next column
    int size1,        // num rows
    unsigned int stride_batch, // stride from one matrix to another
    int size_batch ); // batch size (num matrices)
