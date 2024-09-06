import numpy as np 

def data_array():
    data = np.random.randn(5)
    print(data)

    np.save('datafile.npy', data)

    data = []
    print(data)

    data = np.load('datafile.npy')
    print(data)
    
data_array()
