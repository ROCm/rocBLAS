DEVICE=`lspci |grep -i vega |wc -l`
START_TIME=`date +%m%d_%H%M`
START_TIME_1=`date -u`
ROCLAS_PATH=/$PWD
#variable
declare -i batch_size=8640
declare -x model_name="d"
declare -i loop_time=0
declare -i aver_inters=200
declare -i duration=5

function usage
{
	echo -e "=================================================="
	echo -e " 		AMD Xgemm looping script file"
	echo -e "=================================================="

	echo -e "    -b | --batchsize"
	echo -e "        input Xgemm batch size ex:8640 4096 14096"
	echo -e "    -m | --modelname"
	echo -e "        'd' dGEMM"
	echo -e "        's' sGEMM"
	echo -e "        'h' hGEMM"
#	echo -e "    -t | --looptime"
#	echo -e "        time step in minutes"
        echo -e "    -a | --averinters"
        echo -e "        output average xGEMM result every xx times"
        echo -e "    -d | --duration"
        echo -e "        setting long run times in minute"

}

function kill_Process
{

  until [ $(ps -a |grep -i rocblas |wc -l) == 0 ]
  do
          sleep 1
          ps -a | grep -i rocblas-bench | awk '{print $1}' | xargs -n 1 -i kill {}

  done
}


function wait_Process
{
  sleep 1
  until [ $(ps -a |grep -i rocblas |wc -l) == 0 ]
  do
      sleep 1
      if [[ $(date -u +%s) -gt $STOP_TIME_SECS ]]
      then
          echo "times up!!!"
          kill_Process
      fi
  done
}

function create_file_header
{
    if [ ! -e $RESULT_LOG ]
    then
        touch $RESULT_LOG
    fi

    LINES=`cat ${LOG_NAME}_0.log | grep -i average | wc -l`
    i=0

    echo "times," > $RESULT_LOG
    while [ $i -lt $DEVICE ]
    do
#        sed -i '1s/$/device '$i',/' $RESULT_LOG
        sed -i '1s/$/device '$i' Gflops, date,/' $RESULT_LOG
         (( i++ ))
    done

    i=1
    while [ $i -le $LINES ]
    do
        echo "$i," >> $RESULT_LOG
        (( i++ ))
    done

}


function collect_data
{
    file=$1
    count=2
    cat $file |grep -i average | awk '{print $2 "," $3 }' |
    {
        while IFS= read -r line
        do
            sed -i ''$count's/$/'$line',/' $RESULT_LOG
            (( count++ ))
        done

    }

}




function run_dgemm
{
    i=0
    while [ $i -lt $DEVICE ]
    do
    #    touch $LOG_NAME$i.log
       #/home/h2-hq-01/rocBLAS/build/release/clients/staging/rocblas-bench -f gemm -r d -m 8640 -n 8640 -k 8640 --transposeB T -i 100 --device $i -a 10| tee -a $LOG_NAME_$i.log &
       $ROCLAS_PATH/rocblas-bench -f gemm -r $model_name -m $batch_size  -n $batch_size -k $batch_size --transposeB T -i 10000 --device $i -a $aver_inters| tee -a ${LOG_NAME}_$i.log &
        (( i++ ))
    done
}





while [ "$1" != "" ]; do
    case $1 in
        -b | --batchsize )      shift
								if [ -z $1 ]
								then
									echo "invalid parameter -b"
									exit 1
								else
									batch_size=$1
								fi
                                ;;

        -m | --modelname )      shift
								if [ -z $1 ]
								then
									echo "invalid parameter -m"
									exit 1
								else
									model_name=$1
								fi
								;;

#        -t | --looptime )       shift
#								if [ -z $1 ]
#								then
#									echo "invalid parameter -t"
#									exit 1
#								else
#									loop_time=$1
#								fi
#								;;
#
        -a | --averinters )     shift
								if [ -z $1 ]
								then
									echo "invalid parameter -a"
									exit 1
								else
									aver_inters=$1
								fi
								;;

        -d | --duration )       shift
								if [ -z $1 ]
								then
									echo "invalid parameter -d"
									exit 1
								else
									duration=$1
								fi
								;;

        -h | --help )           usage
                                exit
                                ;;
        * )                     usage
                                exit 1
    esac
    shift
done

trap "kill_Process" SIGINT
LOG_PATH=log/$START_TIME
RESULT_LOG=$LOG_PATH/${model_name}gemm_$DEVICE_$START_TIME.csv
START_TIME_SECS=`date -u +%s`
STOP_TIME_SECS=`date -ud "$duration minutes" +%s`
STOP_TIME=`date -ud "$duration minutes" `
LOG_NAME=$LOG_PATH/${model_name}gemm_$START_TIME

if [ ! -d $LOG_PATH ]
then
    mkdir -p $LOG_PATH
fi

run_dgemm
wait_Process
echo -e "Dgemm process done"
echo -e "parsing data to ${RESULT_LOG}"
create_file_header

i=0
while [ $i -lt $DEVICE ]
do
    collect_data  ${LOG_NAME}_$i.log
    (( i++ ))
done

echo -e "parsing done"



echo -e "\n\n\n" >> $RESULT_LOG
echo -e "==================================================" | tee -a $RESULT_LOG
echo -e " 		AMD Xgemm looping script file" | tee -a $RESULT_LOG
echo -e "==================================================" | tee -a $RESULT_LOG
echo -e "| parameter setting" | tee -a $RESULT_LOG
echo -e "|  batch size    :$batch_size" | tee -a $RESULT_LOG
echo -e "|  model         :${model_name}gemm" | tee -a $RESULT_LOG
echo -e "|  average output:$aver_inters" | tee -a $RESULT_LOG
echo -e "|  test time     :${duration} minutes" | tee -a $RESULT_LOG
echo -e "|  start time    :$START_TIME_1"  | tee -a $RESULT_LOG
echo -e "|  stop time     :$STOP_TIME" | tee -a $RESULT_LOG
echo -e "==================================================" | tee -a $RESULT_LOG
