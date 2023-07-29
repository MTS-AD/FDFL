for dataname in 'HAR' 'CAP' 'SEDFx'
do
  #for alg in 'fedavg' 'fedprox' 'scaffold' 'moon'
  for alg in 'fedavg'
    do
      python FedTranAD.py --dataname $dataname --alg $alg
    done
done