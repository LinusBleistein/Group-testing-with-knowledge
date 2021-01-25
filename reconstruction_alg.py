import numpy as np 
import seaborn as sns 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import cvxpy as cp 

class reconstruction_algorithm():

	def __init__(self,true_x,test_matrix):
		
		self.true_x = true_x
		self.popsize = true_x.shape[0]
		self.test_matrix = test_matrix
		
		if self.true_x.shape[0] != (self.test_matrix.shape)[1]:
			
			raise Exception("Number of columns in the test matrix and true vector size should be equal.")
			
		self.test_result = np.int32(test_matrix@true_x >= 1)

	def accuracy(self,test_mat=None):
				
		if type(test_mat) != type(None):
						
			reconstructed_x = self.reconstruct(test_mat=test_mat)
		
		else:
			
			reconstructed_x = self.reconstruct()
			
			
		confmat = confusion_matrix(self.true_x,reconstructed_x)

		return (confmat[0,0]+confmat[1,1])/np.sum(confmat)

	def precision(self, test_mat=None):

		if type(test_mat) != type(None):
			
			#Allows for reconstruction with different test matrix, used in average_score()
			
			reconstructed_x = self.reconstruct(test_mat=test_mat)
		
		else:
			
			reconstructed_x = self.reconstruct()

		confmat = confusion_matrix(self.true_x,reconstructed_x)

		if confmat[1,1] == confmat[0,1] == confmat[1,0] == 0:

			return 1

		if confmat[1,1] ==0:

			return 0

		else:

			return confmat[1,1]/(confmat[1,1]+confmat[0,1])


	def recall(self, test_mat=None):

		if type(test_mat) != type(None):
			
			#Allows for reconstruction with different test matrix, used in average_score()
			
			reconstructed_x = self.reconstruct(test_mat=test_mat)
		
		else:
			
			reconstructed_x = self.reconstruct()

		confmat = confusion_matrix(self.true_x,reconstructed_x)

		if confmat[1,1] == confmat[1,0] == confmat[0,1] == 0 :

			return 1

		if confmat[1,1] ==0:

			return 0

		else:

			return confmat[1,1]/(confmat[1,1]+confmat[1,0])



	def f1score(self,test_mat=None):

		#confmat = confusion_matrix(self.true_x,self.reconstruct())
		precision = self.precision()
		recall = self.recall()

		if (precision == 0) or (recall==0):

			return 0

		#if confmat[1,1] == confmat[1,0] == confmat[0,1] == 0:

			#return 1

		else:

			#print(self.precision())
			#print(self.recall())

			return 2*(precision*recall)/(precision+recall)

	
	def average_accuracy(self,mat_size,number,alpha):
		
		output = 0
		
		for i in np.arange(number):
			
			random_matrix = np.random.choice([0,1],size=(mat_size,self.popsize),p=[alpha,1-alpha])
			
			output += self.score(test_mat=random_matrix)
			
		return output/number
	
	def confusion_matrix(self,absv = False, plot=False):

		reconstructed_x = self.reconstruct()
		true_x = self.true_x
		confmat = confusion_matrix(true_x,reconstructed_x)

		if plot == False and absv == True :

			return confmat

		if plot == False and absv == False : 

			return confmat/np.sum(confmat)

		if plot == True and absv == False:

			sns.heatmap(confmat/np.sum(confmat),annot=True)
			plt.show()

		if plot == True and absv == True:

			sns.heatmap(confmat,annot=True)
			plt.show()


class COMP(reconstruction_algorithm):
	
	""" 
	
	The COMP algorithm conservatively reconstructs the true vector x given a test matrix and a test result.
	It works by declaring healthy only the individuals that are in a group with negative test result.
	
	Initialize COMP with the true (unknown) vector x and the test matrix. 
	
	"""
	
	def __init__(self,true_x,test_matrix):
		
		super().__init__(true_x, test_matrix)
		
	def reconstruct(self,test_mat=None):
		
		
		if type(test_mat) != type(None):
									
			test_matrix = test_mat
			test_result = test_mat@(self.true_x)
			
		else:
			
			test_matrix = self.test_matrix
			test_result = self.test_result
		
		reconstructed_x = np.ones(test_matrix.shape[1])
		
		for line in np.arange(test_matrix.shape[0]):
			
			if test_result[line] == 0:
				
				def_negatives = test_matrix[line].nonzero()
				
				reconstructed_x[def_negatives] = 0
				
		return reconstructed_x


#######################################################################################################
#######################################################################################################
#######################################################################################################

class DD(reconstruction_algorithm):

	""" 
	
	The DD algorithm reconstructs the true vector x given a test matrix and a test result.
	It works in three steps: 
		- It first declares healthy all individuals that are in a group with negative test result.
		- It then spots the tests where all individuals have been precedently declared healthy, except one, and
		  declares this individual sick.
		- It finally declares all remaining individuals healthy.

	Initialize DD with the true (unknown) vector x and the test matrix. 
	
	"""
	
	def __init__(self,true_x,test_matrix):
		
		super().__init__(true_x, test_matrix)
		
	def reconstruct(self,test_mat=None):
				
		if type(test_mat) != type(None):
									
			test_matrix = test_mat
			test_result = test_mat@(self.true_x)
			
		else:
			
			test_matrix = self.test_matrix
			test_result = self.test_result
		
		nd = np.ones(test_matrix.shape[1])
		
		for line in np.arange(test_matrix.shape[0]):
			
			if test_result[line] == 0:
				
				def_negatives = test_matrix[line].nonzero()
				
				nd[def_negatives] = 0
				
		nd_c_indices = np.nonzero(nd)[0]
		nd_c = 1-nd
		nd_indices = np.nonzero(nd == 0)
		
		defectives = []
		
		for line in np.arange(test_matrix.shape[0]):
						
			test_indices = test_matrix[line].nonzero()[0]
			
			overlap = len(np.intersect1d(test_indices,nd_c_indices))
			
			if overlap == 1:
				
				defective = np.intersect1d(test_indices,nd_c_indices)
				defectives.append(defective[0])
				
		reconstructed_x = np.zeros(test_matrix.shape[1])
		reconstructed_x[defectives]=1
		
		return reconstructed_x

#######################################################################################################
#######################################################################################################
#######################################################################################################

class SCOMP(reconstruction_algorithm):

	""" 
	
	The SCOMP algorithm reconstructs the true vector x given a test matrix and a test result.
	It works in three steps: 
		- It first declares healthy all individuals that are in test groups that test negative. 
		- It then declares sick all individuals that are in positive groups whose members have all been identified
		as negative in the precedent step.

	These two steps mimic the DD algorithm, with the exception the DD algorithm declares healthy all remaining individuals.
	The SCOMP algorithm does not perform this step but performs following steps iteratively:
		- Given an estimate of the true vector x, it checks if it is compatible with the outcome in the sens that testing
		on this vector yields the same outcome as on the true vector. 
		- If not, it identifies the unexplained tests (tests who do not include any confirmed sick individuals) and declares 
		as sick the individual which is included in the greatest number of these tests.  

	Initialize SCOMP with the true (unknown) vector x and the test matrix. 
	
	"""


	def __init__(self,true_x,test_matrix):
		
		super().__init__(true_x, test_matrix)
		
	def reconstruct(self,test_mat=None):
				
		if type(test_mat) != type(None):
									
			test_matrix = test_mat
			test_result = test_mat@(self.true_x)
			
		else:
			
			test_matrix = self.test_matrix
			test_result = self.test_result
		
		nd = np.ones(test_matrix.shape[1])
		
		for line in np.arange(test_matrix.shape[0]):
			
			if test_result[line] == 0:
				
				def_negatives = test_matrix[line].nonzero()
				
				nd[def_negatives] = 0
		
		pd_indices = np.nonzero(nd)[0]
		pd = 1-nd
		nd_indices = np.nonzero(nd == 0)
		
		defectives = []
		
		for line in np.arange(test_matrix.shape[0]):
						
			test_indices = test_matrix[line].nonzero()[0]
			
			overlap = len(np.intersect1d(test_indices,pd_indices))
			
			if overlap == 1:
				
				defective = np.intersect1d(test_indices,pd_indices)
				defectives.append(defective[0])
				
		estimated_x = np.zeros(test_matrix.shape[1])
		estimated_x[defectives] = 1
				
		while np.all(test_matrix@estimated_x == test_matrix@self.true_x) == False :
			
			positive_tests = test_result.nonzero()[0]
			
			unexplained_tests =[]
			
			for test in positive_tests:
								
				if len(np.intersect1d(test_matrix[test].nonzero()[0],estimated_x.nonzero()[0])) == 0:
				
					unexplained_tests.append(test)
					   
			counter = np.zeros(test_matrix.shape[1])
					   
			for i in pd_indices:
					   
			   counter[i] = len(np.intersect1d(test_matrix[:,i].nonzero(),unexplained_tests))
					   
			selected_i = np.random.choice(np.flatnonzero(counter == counter.max()))
			
			if np.all(counter == 0):
				
				return estimated_x
			
			else:
			
				estimated_x[selected_i] = 1
		
		return estimated_x


#######################################################################################################
#######################################################################################################
#######################################################################################################


class LP(reconstruction_algorithm):

	""" 
	
	The LP algorithm reconstructs the true vector x given a test matrix and a test result.
	It works by solving a convex optimization problem interpreted as a relaxation of the true underlying 
	optimisation problem of group testing. The problem is solved using CVXPY.
	For more details, see [Malioutov and Malyutov, 2015]. 
	
	"""
	
	def __init__(self,true_x,test_matrix):
		
		super().__init__(true_x, test_matrix)
		
		
	def reconstruct(self,test_mat=None):
		
		
		if type(test_mat) != type(None):
						
			#Allows for reconstruction with different test matrix, used in average_score()
			
			test_matrix = test_mat
			test_result = test_mat@(self.true_x)
			
		else:
			
			test_matrix = self.test_matrix
			test_result = self.test_result
		
		positive_tests = test_matrix[test_result != 0,:]
		negative_tests = test_matrix[test_result == 0,:]

		x = cp.Variable(self.popsize)
		objective = cp.Minimize(cp.sum(x))

		if len(negative_tests)==0:

			constraints = [ 0 <= x,test_result[test_result != 0] <= positive_tests@x]
			prob = cp.Problem(objective,constraints)
			prob.solve()
		
			optimal_x = np.float32(x.value)
				 
			optimal_x[optimal_x < 1] = 0
		
			return optimal_x

		if len(positive_tests) ==0:

			constraints = [ 0 <= x, x <= 1, negative_tests@x == 0]
			prob = cp.Problem(objective,constraints)
			prob.solve()
		
			optimal_x = np.float32(x.value)
				 
			optimal_x[optimal_x < 1] = 0
		
			return optimal_x

		else:

			constraints = [ 0 <= x, negative_tests@x == 0,test_result[test_result != 0] <= positive_tests@x]
			prob = cp.Problem(objective,constraints)
			prob.solve()
			
			optimal_x = np.float32(x.value)
					 
			optimal_x[optimal_x < 1] = 0
			
			return optimal_x

#######################################################################################################
#######################################################################################################
#######################################################################################################

class Enhanced_LP(reconstruction_algorithm):

	""" 
	
	The Enhanced LP algorithm is similar to the LP algorithm, but it preprocesses the data using the two first steps
	of the DD algorith. In the first stage, it isolates items that are definetly negative and positive using the procedure
	of the DD algorithm. The remaining individuals are categorized as indefinite, and are labeled through convex relaxation. 
	
	"""
	
	def __init__(self,true_x,test_matrix):
		
		super().__init__(true_x, test_matrix)
		
		
	def reconstruct(self,test_mat=None):
		
		
		if type(test_mat) != type(None):
						
			#Allows for reconstruction with different test matrix, used in average_score()
			
			test_matrix = test_mat
			test_result = test_mat@(self.true_x)
			
		else:
			
			test_matrix = self.test_matrix
			test_result = self.test_result

		nd = np.ones(test_matrix.shape[1])
		
		for line in np.arange(test_matrix.shape[0]):
			
			if test_result[line] == 0:
				
				def_negatives = test_matrix[line].nonzero()
				
				nd[def_negatives] = 0
				
		nd_c_indices = np.nonzero(nd)[0]
		nd_c = 1-nd
		nd_indices = np.nonzero(nd == 0)
		
		defectives = []
		
		for line in np.arange(test_matrix.shape[0]):
						
			test_indices = test_matrix[line].nonzero()[0]
			
			overlap = len(np.intersect1d(test_indices,nd_c_indices))
			
			if overlap == 1:
				
				defective = np.intersect1d(test_indices,nd_c_indices)
				defectives.append(defective[0])
				
		reconstructed_x = np.zeros(test_matrix.shape[1])
		reconstructed_x[defectives]=1
		
		return reconstructed_x
		
		positive_tests = test_matrix[test_result != 0,:]
		negative_tests = test_matrix[test_result == 0,:]

		x = cp.Variable(self.popsize)
		objective = cp.Minimize(cp.sum(x))

		if len(negative_tests)==0:

			constraints = [ 0 <= x, x <= 1,test_result[test_result != 0] <= positive_tests@x]
			prob = cp.Problem(objective,constraints)
			prob.solve()
		
			optimal_x = x.value
				 
			optimal_x[optimal_x < 1] = 0
		
			return optimal_x.astype(int)

		if len(positive_tests) ==0:

			constraints = [ 0 <= x, x <= 1, negative_tests@x == 0]
			prob = cp.Problem(objective,constraints)
			prob.solve()
		
			optimal_x = x.value
				 
			optimal_x[optimal_x < 1] = 0
		
			return optimal_x.astype(int)

		else:

			constraints = [ 0 <= x, x <= 1, negative_tests@x == 0,test_result[test_result != 0] <= positive_tests@x]
			prob = cp.Problem(objective,constraints)
			prob.solve()
			
			optimal_x = x.value
					 
			optimal_x[optimal_x < 1] = 0
			
			return optimal_x.astype(int)




