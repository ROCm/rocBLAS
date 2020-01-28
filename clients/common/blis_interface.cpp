#include "blis.h"
#include "omp.h"

void setup_blis()
{
    bli_init();
}

static int initialize_blis = (setup_blis(), 0);
