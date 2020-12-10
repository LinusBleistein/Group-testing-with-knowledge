import numpy as np 

def create_input(n,p):
	"""
	Generates a random input vector with coordinates that are Bernouilli random variables with sickness (x_i = 1) probability 1-p.
	"""
	
	return np.random.choice([0,1],size=n,p=[p,1-p])

def create_tests(T,n,alpha):
	"""
	Generates a random test matrix with T tests, designed for an input vector of size n. Every entry is randmly set to 1 with probability 1-alpha.
	""" 

	return np.random.choice([0,1],size=(T,n),p=[alpha,1-alpha])