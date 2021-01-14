import numpy as np 

def vsparse_input(n,k,p,iid=False):
	"""
	Generates a random very sparse input vector of size n. If iid is set to True, generates a vector with a iid prior, with probability of being defective equal to p. 
	Otherwise, the vector is generated on the combinatorial prior with k defectives. 
	"""
	
	if iid:

		return np.random.choice([0,1],size=n,p=[1-p,p])

	else:

		defectives = np.random.randint(0,n,size=k)

		vector = np.zeros(n)

		vector[defectives] = 1

		return vector 

def sparse_input(n,alpha):
	"""
	Generates a random sparse input vector of size n. Alpha controles the sparsness of the vector: in total, floor(n**alpha) entries choosen uniformly will be set to 1.
	Alpha should be between 0 and 1.
	"""

	ndefectives = int(n**alpha)

	defectives = np.random.randint(0,n,size=ndefectives)

	vector = np.zeros(n)

	vector[defectives] = 1

	return vector

def linear_input(n,beta):
	"""
	Generates a random "linear-sparse" vector. beta*n coefficients are choosen uniformly and set to 1. beta should be bewteen 0 and 1.  
	"""

	ndefectives = int(beta*n)

	defectives = np.random.randint(0,n,size=ndefectives)

	vector = np.zeros(n)

	vector[defectives] = 1

	return vector

def b_tests(T,n,p):
	"""
	Generates a random test matrix with T tests, designed for an input vector of size n. Every entry is randmly set to 1 with probability 1-alpha.
	""" 

	return np.random.choice([0,1],size=(T,n),p=[1-p,p])


def ctpi_tests(T,n,L):
	
	"""
	Generates a random test matrix with T tests, designed for an input vector of size n. Entries are designed such that every item is tested L times.
	"""
	
	testmat = np.zeros((T,n))
	aux = np.random.random((T,n))
	testmat[aux.argsort(axis=0)>= T-L] = 1
    
	return np.int64(testmat)


