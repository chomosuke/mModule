import pandas as pd
import numpy as np

@np.vectorize
def sigmoid(x):
	if x > 500:
		return 1
	if x < -500:
		return 0
	return 1 / (1 + np.e ** -x)

from scipy.stats import truncnorm
def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

class NeuralNetwork:
        
	def __init__(self, nos_of_nodes):  

		self.nos_of_nodes = nos_of_nodes

		# Initialize the weight matrices of the neural network with optional bias nodes
		self.weight_matrices = []
		for i in range(len(self.nos_of_nodes) - 1):
			rad = 1 / np.sqrt(self.nos_of_nodes[i] + 1)
			X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
			# append 2d np arrays representing weight matrices
			self.weight_matrices.append(X.rvs((self.nos_of_nodes[i+1], 
			                               self.nos_of_nodes[i] + 1)))

	def train(self, input_vectors, target_vectors, learning_rate):
		average_delta = []
		for weight_matrice in self.weight_matrices:
			average_delta.append(np.zeros(weight_matrice.shape))

		for i in range(len(input_vectors)):
			input_vector = input_vectors[i]
			target_vector = target_vectors[i]
			# input_vector and target_vector can be tuple, list or ndarray
			input_vector = np.array(input_vector, ndmin=2)
			input_vector = input_vector.T

			output_vectors = [input_vector]
			# adding bias node
			output_vectors[-1] = np.concatenate( (output_vectors[-1], [[1]]) )
			for weight_matrice in self.weight_matrices:
				vector_tmp = output_vectors[-1]
				vector_tmp = np.dot(weight_matrice, vector_tmp)
				vector_tmp = sigmoid(vector_tmp)
				output_vectors.append(vector_tmp)
				# adding bias node
				output_vectors[-1] = np.concatenate( (output_vectors[-1], [[1]]) )
					
					
			target_vector = np.array(target_vector, ndmin=2).T
			target_vector = np.concatenate( (target_vector, [[1]]) )
			
			# calculate errors:
			output_errors = [target_vector - output_vectors[-1]]
			for i2 in range(1, len(self.weight_matrices)):
				output_errors.append(np.dot(self.weight_matrices[-i2].T, output_errors[-1][:-1,:]))
	
			# update the weights:
			for i2 in range(len(output_errors)):
				tmp = output_errors[i2] * output_vectors[-1-i2] * (1.0 - output_vectors[-1-i2])     
				tmp = np.dot(tmp[:-1,:], output_vectors[-2-i2].T)
				average_delta[-1-i2] = average_delta[-1-i2] /(i+1)*i + learning_rate * tmp /(i+1)
		for i in range(len(self.weight_matrices)):
			self.weight_matrices[i] += average_delta[i]
			
	def run(self, input_vector):
		# input_vector can be tuple, list or ndarray
		output_vector = np.array(input_vector, ndmin=2).T

		for weight_matrices in self.weight_matrices:
			# adding bias node
			output_vector = np.concatenate( (output_vector, [[1]]) )
			output_vector = np.dot(weight_matrices, output_vector)
			output_vector = sigmoid(output_vector)
		
		return output_vector
