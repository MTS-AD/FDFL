import h5py
import os

DATA_PATH = '/remote-home/heyf/FDAD_data/CAP.h5'
f= h5py.File(DATA_PATH,'r')
domain=f.keys()
for index,i in enumerate(domain):
    data=f[i]['data'][...]
    label=f[i]['labels'][...]
    print(data.shape,label.shape)

DATA_PATH = '/remote-home/heyf/FDAD_data/SEDFx.h5'
f= h5py.File(DATA_PATH,'r')
domain=f.keys()
for index,i in enumerate(domain):
    data=f[i]['data'][...]
    label=f[i]['labels'][...]
    print(data.shape,label.shape)

import h5py
import os
DATA_PATH = '/remote-home/heyf/FDAD_data/HAR.h5'
f= h5py.File(DATA_PATH,'r')
domain=f.keys()
for index,i in enumerate(domain):
    data=f[i]['data'][...]
    label=f[i]['labels'][...]
    print(data.shape,label.shape)



