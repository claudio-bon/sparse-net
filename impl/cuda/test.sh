FOLDER="test_results/"

R=3
N0=(1000 2500 5000 7500 10000 20000 30000 40000 50000 75000 100000 125000 150000 175000 200000)
LAYERS=(100 250 500)
TRIALS=5

rm -f ${FOLDER}measures_cuda.csv
touch ${FOLDER}measures_cuda.csv

echo "N,K,kt,nops,tot" >> ${FOLDER}measures_cuda.csv

for K in ${LAYERS[@]}; do
  for N in ${N0[@]}; do
    if [[ "$N" -le "$K * ($R - 1)" ]]; then
      continue
    fi
    echo "Input size = $N, Num layers = $K"
    for rep in `seq $TRIALS`; do
      OUT="$(./nn_cuda -n $N -k $K)"
      KERNEL_TIME="$(echo "$OUT" | sed -n 's/^.*Kernel time:\s*\([0-9]\+\.[0-9]\+\).*$/\1/p')"
      NOPS="$(echo "$OUT" | sed -n 's/^.*Nops:\s*\([0-9]\+\).*$/\1/p')"
      TOTAL_TIME="$(echo "$OUT" | sed -n 's/^.*Total time:\s*\([0-9]\+\.[0-9]\+\).*$/\1/p')"

      echo "$rep/$TRIALS:"
      echo -e "\tKernel time: ${KERNEL_TIME}s"
      echo -e "\tNops: $NOPS"
      echo -e "\tTotal time: ${TOTAL_TIME}s"

      echo -n "$N,$K," >> ${FOLDER}measures_cuda.csv
      echo -n "${KERNEL_TIME}," >> ${FOLDER}measures_cuda.csv
      echo -n "${NOPS}," >> ${FOLDER}measures_cuda.csv
      echo -n "${TOTAL_TIME}" >> ${FOLDER}measures_cuda.csv
      echo "" >> ${FOLDER}measures_cuda.csv
    done
    echo ""
  done
done