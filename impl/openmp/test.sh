FOLDER="test_results/"

CORES=`cat /proc/cpuinfo | grep processor | wc -l` # number of cores
R=3



echo "--------- Elapsed Time ---------"

N0=50000
LAYERS=(250)
PARALLELISM=("OUTER" "INNER" "SEQUENTIAL")
TRIALS=5

rm -f ${FOLDER}elapsed_time.csv
touch ${FOLDER}elapsed_time.csv

echo "p,N,K,pt,t" >> ${FOLDER}elapsed_time.csv

for K in ${LAYERS[@]}; do
  if [[ "$N0" -le "$K * ($R - 1)" ]]; then
    continue
  fi
  for PT in ${PARALLELISM[@]}; do
    for p in `seq $CORES`; do
      echo "Num threads = $p, Input size = $N0, Num layers = $K, parallelism = $PT"
      for rep in `seq $TRIALS`; do
        EXEC_TIME="$(OMP_NUM_THREADS=$p ./nn_openmp -n $N0 -k $K -pt $PT | grep -Eo '[0-9]+\.[0-9]+' )"
        echo "Execution time $rep/$TRIALS: ${EXEC_TIME}s"

        echo -n "$p,$N0,$K,$PT," >> ${FOLDER}elapsed_time.csv
        echo -n "${EXEC_TIME}" >> ${FOLDER}elapsed_time.csv
        echo "" >> ${FOLDER}elapsed_time.csv
      done
      echo ""
    done
  done
done


echo "--------- Strong Scaling Efficiency ---------"

N0=50000
LAYERS=(100 250 500)
PT="OUTER"
TRIALS=5

rm -f ${FOLDER}strong_scale.csv
touch ${FOLDER}strong_scale.csv

echo "p,N,K,pt,t" >> ${FOLDER}strong_scale.csv

for K in ${LAYERS[@]}; do
  if [[ "$N0" -le "$K * ($R - 1)" ]]; then
    continue
  fi
  for p in `seq $CORES`; do
    echo "Num threads = $p, Input size = $N0, Num layers = $K, parallelism = $PT"
    for rep in `seq $TRIALS`; do
      EXEC_TIME="$(OMP_NUM_THREADS=$p ./nn_openmp -n $N0 -k $K -pt $PT | grep -Eo '[0-9]+\.[0-9]+' )"
      echo "Execution time $rep/$TRIALS: ${EXEC_TIME}s"

      echo -n "$p,$N0,$K,$PT," >> ${FOLDER}strong_scale.csv
      echo -n "${EXEC_TIME}" >> ${FOLDER}strong_scale.csv
      echo "" >> ${FOLDER}strong_scale.csv
    done
    echo ""
  done
done


echo ""
echo "--------- Weak Scaling Efficiency ---------"

N0=5000
LAYERS=(100 250 500)
PT="OUTER"
TRIALS=5

rm -f ${FOLDER}weak_scale.csv
touch ${FOLDER}weak_scale.csv

echo "p,N,K,pt,t" >> ${FOLDER}weak_scale.csv

for K in ${LAYERS[@]}; do
  for p in `seq $CORES`; do
    # Compute scaled N
    Np=$(( p * N0 + (1-p) * (R-1) * (K/2) ))
    if [[ "$Np" -le "$K * ($R - 1)" ]]; then
      continue
    fi

    echo "Num threads = $p, Input size = $Np, Num layers = $K, parallelism = $PT"
    for rep in `seq $TRIALS`; do
      EXEC_TIME="$(OMP_NUM_THREADS=$p ./nn_openmp -n $Np -k $K -pt $PT | grep -Eo '[0-9]+\.[0-9]+' )"
      echo "Execution time $rep/$TRIALS: ${EXEC_TIME}s"

      echo -n "$p,$Np,$K,$PT," >> ${FOLDER}weak_scale.csv
      echo -n "${EXEC_TIME}" >> ${FOLDER}weak_scale.csv
      echo "" >> ${FOLDER}weak_scale.csv
    done
    echo ""
  done
done


echo ""
echo "--------- Other Measures ---------"

N0=(1000 2500 5000 7500 10000 20000 30000 40000 50000 75000 100000 125000 150000 175000 200000)
LAYERS=(100 250 500)
PT="SEQUENTIAL"
TRIALS=5

rm -f ${FOLDER}measures_openmp.csv
touch ${FOLDER}measures_openmp.csv

echo "N,K,pt,t" >> ${FOLDER}measures_openmp.csv

for K in ${LAYERS[@]}; do
  for N in ${N0[@]}; do
    if [[ "$N" -le "$K * ($R - 1)" ]]; then
      continue
    fi
    echo "Input size = $N, Num layers = $K, parallelism = $PT"
    for rep in `seq $TRIALS`; do
      EXEC_TIME="$(OMP_NUM_THREADS=1 ./nn_openmp -n $N -k $K -pt $PT | grep -Eo '[0-9]+\.[0-9]+' )"
      echo "Execution time $rep/$TRIALS: ${EXEC_TIME}s"

      echo -n "$N,$K,$PT," >> ${FOLDER}measures_openmp.csv
      echo -n "${EXEC_TIME}" >> ${FOLDER}measures_openmp.csv
      echo "" >> ${FOLDER}measures_openmp.csv
    done
    echo ""
  done
done