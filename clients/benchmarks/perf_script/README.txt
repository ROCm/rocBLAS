
Instruction to run the script to collect performance data

1) copy client.exe (Windows, client on Linux) to this folder 

2) "python measurePerformance.py --help" to see how to run the command line mode
    For example, "python measurePerformance.py -s 64-5760:64 -f gemm -r s --transa none --transb transpose > sgemm.txt" 
    will collect performance of sgemm NT of square matrices [64:5760] with step 64.
    All the output will be dumped into sgemm.txt

3) Open the sgemm.txt and use "grep" to fetch the performance line. You can extract the Gflop/s number from the performance line

4) You can ingore the other automatic generated .txt files and folders during the run. 
