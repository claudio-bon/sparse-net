# -*- coding: utf-8 -*-

import numpy as np
import argparse
from os import path

def create_nn(N,K,R,filename="datafile",file_path="_data"):
    
    if N-(K-1)*(R-1) <= 0:
        raise ValueError("Either N is too small or K is too big")
        
        
    filename = filename+"_N{}_K{}_R{}.txt".format(N,K,R)
    with open(path.join(file_path,filename), 'w+') as f:
	
        f.write(str(K)+'\n')
		
        for id_layer in range(1,K):
            layer_dim = N-id_layer*(R-1)
				
            np.savetxt(f, np.random.normal(size=(layer_dim,R)),
					   delimiter=' ',footer='', comments='',
                       header='{}\n{}\n{}'.format(layer_dim,R,np.random.normal(size=())))



def create_vector(N,filename="vector",file_path="data"):
	filename = filename+"_N{}.txt".format(N)
	with open(path.join(file_path,filename), 'w+') as f:
		np.savetxt(f, np.random.normal(size=N),
                   delimiter=' ',footer='', comments='',
                   header='{}'.format(N))



def main(N,K,R):
    create_nn(N,K,R)
    create_vector(N)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Generate neural network's layers and input")
	parser.add_argument("-n", type=int, required=True, metavar="N", dest="N")
	parser.add_argument("-k", type=int, required=True, metavar="K", dest="K")
	parser.add_argument("-r", type=int, required=True, metavar="R", dest="R")
	args = parser.parse_args()
	N = args.N
	K = args.K
	R = args.R
	main(N,K,R)