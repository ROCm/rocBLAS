/Version:1
//Writer: simon.huang@amd.com
//rocblas_excutor.sh


1. Paramater setting.
==================================================
            AMD Xgemm looping script file
==================================================
    -b | --batchsize
        input Xgemm batch size ex:8460 4096 14096
    -m | --modelname
        'd' dGEMM
        's' sGEMM
        'h' hGEMM
    -a | --averinters
        output average xGEMM result every xx times
    -d | --duration
        setting long run times in minute

2. Enviorment setting
    vi rocblas_excutor.sh

    ROCLAS_PATH: to your rocblas-bench location path.
    LOG_PATH:

3. Output log files


    1."x"gemm_$date_$device.csv
    2. each device's performance .log

//sample
#rocblas_excutor.sh -b 8640 -m d -a 2 -d 37
#rocblas_excutor.sh -b 4096 -m s -a 5 -d 5

after process done, will output 2 file type in LOG_PATH
