How to run the script to collect performance data

1) copy client.exe (Windows, client on Linux) to this folder 
2) python measurePerformance.py -s 32-5760:32 -f gemm -r -s --transposeA N --transposeB T --tablefile sgemm.csv 


will collect performance of sgemm NT of square matrices [32:5760] with step 32.
The output file is sgemm.csv

