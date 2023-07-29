for dataname in 'SEDFx'
do
  #for alg in 'fedavg' 'fedprox' 'scaffold' 'moon'
  for alg in 'scaffold':
    do
      python FedLSTM_AE.py --dataname $dataname --alg $alg
    done
done