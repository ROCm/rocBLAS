
CobaltTensor initializeTensorForGEMM(
    CobaltTensor & tensor,
    CobaltDataType dataType,
    int stride0,      // stride from one row to another
    int size0,        // num columns
    int stride1,      // stride from one column to next column
    int size1,        // num rows
    unsigned int strideBatch, // stride from one matrix to another
    int sizeBatch );  // batch size (num matrices)
