for dataname in 'HAR' 'CAP' 'SEDFx'
do
  #for alg in 'fedavg' 'fedprox' 'scaffold' 'moon'
  for alg in 'scaffold'
    do
      python FedGDN.py --dataname $dataname --alg $alg
    done
done