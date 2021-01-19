import numpy as np 
import pandas as pd
from sklearn.datasets import make_blobs,make_moons
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


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

def generate_blobs(samplesize=500,sparsity_regime="sparse",sparsity_parameter=2/3):

	if sparsity_regime == "sparse":

		n_defectives = int(samplesize**(sparsity_parameter))

	if sparsity_regime == "vsparse":

		n_defectives = sparsity_parameter

	if sparsity_regime == "linear":

		n_defectives = sample_size*sparsity_parameter

	X, y = make_blobs(n_samples=[samplesize,n_defectives], n_features=2, random_state=0, centers=np.array([[0,0],[1,-1]]))

	x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

	clf = LogisticRegression().fit(x_train, y_train)

	probas = clf.predict_proba(x_test)[:,0]

	predicted = clf.predict(x_test)

	length = len(x_test)

	data = np.concatenate((y_test.reshape(length,1),x_test,probas.reshape(length,1),predicted.reshape(length,1),predicted.reshape(length,1)),axis=1)

	data = data.reshape(length,6)

	df = pd.DataFrame(data,columns=['True status','Feature 1','Feature 2','Predicted probability','Predicted status','Rectified status'])

	return df 



def generate_moons(samplesize=500,sparsity_regime="sparse",sparsity_parameter=2/3,neighbours=10):

	if sparsity_regime == "sparse":

		n_defectives = int(samplesize**(sparsity_parameter))

	if sparsity_regime == "vsparse":

		n_defectives = sparsity_parameter

	if sparsity_regime == "linear":

		n_defectives = samplesize*sparsity_parameter

	X, y = make_moons(n_samples=(400,60), noise=.45)

	x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

	knn = KNeighborsClassifier(n_neighbors=neighbours)
	knn.fit(x_train, y_train)

	probas = knn.predict_proba(x_test)[:,0]

	predicted = knn.predict(x_test)

	length = len(x_test)

	data = np.concatenate((y_test.reshape(length,1),x_test,probas.reshape(length,1),predicted.reshape(length,1),predicted.reshape(length,1)),axis=1)

	data = data.reshape(length,6)

	df = pd.DataFrame(data,columns=['True status','Feature 1','Feature 2','Predicted probability','Predicted status','Rectified status'])

	return df 
















