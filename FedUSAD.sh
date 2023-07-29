for dataname in 'HAR' 'CAP' 'SEDFx'
do
  #for alg in 'fedavg' 'fedprox' 'scaffold' 'moon'
  for alg in 'scaffold'
    do
      python FedUSAD.py --dataname $dataname --alg $alg
    done
done