How to run the script to collect performance data

e.g 

python measurePerformance.py -s 32-5760:32 -f gemm -r -s --transposeA N --transposeB T --tablefile sgemm.csv 


will collect performance of sgemm NT of square matrices [32:5760] with step 32.
The output file is sgemm.csv

