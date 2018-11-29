import numpy as np
import matplotlib.pyplot as plt
import math

class nn():

	def __init__(self):
		self.hidden_layers = None
		self.input_nodes = None
		self.output_nodes = None
		self.avg_no = None
		self.weights_list = []
		self.bias_list = []
		self.grad = None
		self.lr = 0.
		self.C = 0.

	def set_params(self,hl,input_nodes, output, lr = 0.05,C = 0.1, avg_no = 40):
		self.hidden_layers = hl
		self.input_nodes = input_nodes
		self.output_nodes = output
		self.lr = lr
		self.C = C
		self.avg_no = avg_no
		self.grad = np.zeros(self.avg_no)

		self.set_weights_bias()

	def set_weights_bias(self):
		for i in range(len(self.hidden_layers)+1):
			print(i)
			if i == 0:
				self.weights_list.append(np.matrix(self.weights(size = (self.input_nodes,self.hidden_layers[i]))))
				self.bias_list.append(np.matrix(self.biases(size = (1,self.hidden_layers[i])))) #1x40
			
			elif i == len(self.hidden_layers):
				self.weights_list.append(np.matrix(self.weights(size = (self.hidden_layers[i-1],self.output_nodes)))) #20x1
				self.bias_list.append(np.matrix(self.biases(size = (1,self.output_nodes))))
			
			else:
				self.weights_list.append(np.matrix(self.weights(size = (self.hidden_layers[i-1],self.hidden_layers[i])))) #40x20
				self.bias_list.append(np.matrix(self.biases(size = (1,self.hidden_layers[i])))) #1x20


	def forward_pass(self,data):
		logits = []
		activated_layers = []

		for i in range(len(self.weights_list)):
			if i == 0:
				logits.append(np.add(np.matmul(data,self.weights_list[i]),self.bias_list[i]))
				activated_layers.append(self.tanh(logits[i]))

			elif i == (len(self.weights_list) - 1):
				logits.append(np.add(np.matmul(activated_layers[i-1],self.weights_list[i]),self.bias_list[i]))
				activated_layers.append(self.linear(logits[i]))

			else:
				logits.append(np.add(np.matmul(activated_layers[i-1],self.weights_list[i]),self.bias_list[i]))
				activated_layers.append(self.tanh(logits[i]))
			
		return activated_layers

	def train(self, data, actual_y, total_epoch = 5000, threshold = 0.000001):
		activated_layer = None
		
		data_points = len(actual_y)


		loss_graph = []
		error = 1.
		current_epoch = 0
		dweights_moment, dbias_moment = [0.]*len(self.weights_list), [0.]*len(self.weights_list)
		#threshold = 0.0000001

		print(range(len(self.weights_list)-1,-1,-1))
		while error > threshold and current_epoch<total_epoch:
			current_epoch +=1
			activated_layer = self.forward_pass(data)
			dweights, dbias = [], []
			for i in xrange(len(self.weights_list)-1,-1,-1):
				if i == len(self.weights_list)-1:
					base_weights = np.multiply(self.loss_func_mse_der(actual_y,activated_layer[-1])/data_points,self.der_linear(activated_layer[i]))
					base_bias = np.multiply(self.loss_func_mse_der(actual_y,activated_layer[-1])/data_points,self.der_linear(activated_layer[i]))
					
					dweights.append(np.dot(activated_layer[i-1].T,base_weights) +  (self.C/data_points)*self.weights_list[i]) 
					dbias.append(base_bias.sum(axis = 0))
					
					if current_epoch == 1:
						dweights_moment[len(self.weights_list) - i - 1] = self.ewa([], 0.9, dweights[len(self.weights_list) - i - 1])
						dbias_moment[len(self.weights_list) - i - 1] = self.ewa([], 0.9, dbias[len(self.weights_list) - i - 1])
					else:
						dweights_moment[len(self.weights_list) - i - 1] = self.ewa(dweights_moment[len(self.weights_list) - i - 1], 0.9, dweights[len(self.weights_list) - i - 1])
						dbias_moment[len(self.weights_list) - i - 1] = self.ewa(dbias_moment[len(self.weights_list) - i - 1], 0.9, dbias[len(self.weights_list) - i - 1])

				elif i == 0:
					base_weights = np.multiply(np.dot(base_weights,self.weights_list[i+1].T),self.der_tanh(activated_layer[i]))
					dweights.append(np.dot(data.T,base_weights) + (self.C/data_points)*self.weights_list[i]) 

					base_bias = np.multiply(np.dot(base_bias,self.weights_list[i+1].T),self.der_tanh(activated_layer[i]))
					dbias.append(base_bias.sum(axis = 0))

					if current_epoch == 1:
						dweights_moment[len(self.weights_list) - i - 1] = self.ewa([], 0.9, dweights[len(self.weights_list) - i - 1])
						dbias_moment[len(self.weights_list) - i - 1] = self.ewa([], 0.9, dbias[len(self.weights_list) - i - 1])
					else:
						dweights_moment[len(self.weights_list) - i - 1] = self.ewa(dweights_moment[len(self.weights_list) - i - 1], 0.9, dweights[len(self.weights_list) - i - 1])
						dbias_moment[len(self.weights_list) - i - 1] = self.ewa(dbias_moment[len(self.weights_list) - i - 1], 0.9, dbias[len(self.weights_list) - i - 1])

				elif (i>0 and i <len(self.weights_list)-1):
					base_weights = np.multiply((np.dot(base_weights,self.weights_list[i+1].T)),self.der_tanh(activated_layer[i]))
					dweights.append(np.dot(activated_layer[i-1].T,base_weights) + (self.C/data_points)*self.weights_list[i])
				
					base_bias = np.multiply((np.dot(base_bias,self.weights_list[i+1].T)),self.der_tanh(activated_layer[i]))
					dbias.append(base_bias.sum(axis = 0))

					if current_epoch == 1:
						dweights_moment[len(self.weights_list) - i - 1] = self.ewa([], 0.9, dweights[len(self.weights_list) - i - 1])
						dbias_moment[len(self.weights_list) - i - 1] = self.ewa([], 0.9, dbias[len(self.weights_list) - i - 1])
					else:
						dweights_moment[len(self.weights_list) - i - 1] = self.ewa(dweights_moment[len(self.weights_list) - i - 1], 0.9, dweights[len(self.weights_list) - i - 1])
						dbias_moment[len(self.weights_list) - i - 1] = self.ewa(dbias_moment[len(self.weights_list) - i - 1], 0.9, dbias[len(self.weights_list) - i - 1])

			for i in range(len(self.weights_list)-1,-1,-1):
				self.weights_list[i] = self.weights_list[i] - self.lr*dweights_moment[len(self.weights_list) - i - 1]
				self.bias_list[i] = self.bias_list[i] - self.lr*dbias_moment[len(self.weights_list) - i - 1]

			error = self.loss_func_mse(actual_y,activated_layer[-1])/data_points

			if current_epoch%10000 == 0:
				loss_graph.append(error)
				print('Error at {}:{}'.format(current_epoch,error))

		return loss_graph, activated_layer[-1]

	@staticmethod
	def ewa(moving_avg, beta, current_weights):
		if len(moving_avg) == 0:
			return (1-beta)*current_weights
		else:
			momentum = beta*(moving_avg) + (1 - beta)*current_weights
			return momentum

	def show_weights_bias(self):
		print('Weights: {}'.format(self.weights_list))
		print('Bias: {}'.format(self.bias_list))

	def predict(self,test_data):
		activated_layer = self.forward_pass(test_data)

		return activated_layer[-1]

	@staticmethod
	def weights(mean = 0.0, std = 1.0 ,size=(0,0)):
		return np.random.normal(mean,std,size)

	@staticmethod
	def biases(mean = 0.0, std = 0.0 ,size=(0,0)):
		return np.random.normal(mean,std,size)

	@staticmethod
	def loss_func_mse(actual_y,predicted_y):
		return np.matrix.sum(np.power((predicted_y - actual_y),2))

	@staticmethod
	def loss_func_mse_der(actual_y,predicted_y):
		return 2*(predicted_y-actual_y)

	@staticmethod
	def tanh(x):
		return (2/(1+np.exp(-2*x))) - 1

	@staticmethod
	def der_tanh(x):
		return 1 - np.power(x,2)

	@staticmethod
	def der_sigmoid(x):
		return np.multiply(x,(1.0-x))

	@staticmethod
	def sigmoid(x):
		return 1.0/(1.0+ np.exp(-x))

	@staticmethod
	def linear(x):
		return x

	@staticmethod
	def der_linear(x):
		return np.ones_like(x)


def max_min_transform(arr):
	a = (arr.max(axis = 0)-arr.min(axis = 0))
	b = arr.min(axis = 0)
	arr = (arr - b)/a	

	print(arr.shape, a.shape, b.shape)
	return arr,a,b

def standardization(data, mean_data, std_data):
	data = (data - mean_data)/std_data
	#print(mean_data,std_data)
	return data

if __name__ == '__main__':

	theta1 =np.linspace(-np.pi/4.,np.pi/4.,250)
	theta2 = np.linspace(-np.pi/4.,np.pi/4.,250)
	
	np.random.seed(10)
	np.random.shuffle(theta1)
	np.random.shuffle(theta2)
	
	xp = np.cos(theta1) + np.cos(theta1 + theta2)
	yp = np.sin(theta1) + np.sin(theta1 + theta2)
	#xp = np.cos(theta1)
	#yp = np.sin(theta1)
	dataset = np.array([xp, yp, theta1,theta2]).T
	np.random.shuffle(dataset)
	dataset = np.matrix(dataset)

	actual_y = dataset[:,2:]
	print(actual_y.shape)
	#actual_y,a,b = max_min_transform(actual_y)
	
	data = dataset[:,:2]
	data_std = standardization(data, data.mean(axis = 0), data.std(axis =0))

	my_nn = nn()
	my_nn.set_params([3,5,4],2,2,lr = 0.05,C = 0.1, avg_no = 40)
	loss_graph, train_predict = my_nn.train(data_std,actual_y,total_epoch = 60000, threshold = 0.00000000001)
	plt.plot(range(len(loss_graph)), loss_graph)
	plt.show()

	##### Testing New Data #########
	
	# Case 1:
	'''
	test_theta1 = np.linspace(0,np.pi/2.,22)
	test_theta2 = np.linspace(0,np.pi/4.,22)
	test_xp = np.cos(test_theta1) + np.cos(test_theta1 + test_theta2)
	test_yp = np.sin(test_theta1) + np.sin(test_theta1 + test_theta2)
	'''

	# Case 2:
	test_theta1 = np.linspace(0,np.pi/4.,5)
	test_theta2 = np.linspace(-np.pi/4.,0,5)
	np.random.shuffle(test_theta2)
	np.random.shuffle(test_theta1)
	test_xp = np.cos(test_theta1) + np.cos(test_theta1 + test_theta2)
	test_yp = np.sin(test_theta1) + np.sin(test_theta1 + test_theta2)

	test_dataset = np.array([test_xp, test_yp]).T
	test_dataset = np.matrix(test_dataset)

	#test_y = (test_dataset[:,2:] - b)/a
	test_data = standardization(test_dataset[:,:2], data.mean(axis = 0), data.std(axis =0))
	test_predict = my_nn.predict(test_data)
	theta_predict = test_predict

	# Get the values of theta 1 and theta 2
	#theta_predict = np.multiply(test_predict,a) + b

	predicted_x = np.cos(theta_predict[:,0]) + np.cos(theta_predict[:,0] + theta_predict[:,1])
	predicted_y = np.sin(theta_predict[:,0]) + np.sin(theta_predict[:,0] + theta_predict[:,1])

	loss = math.sqrt(np.sum(np.power((test_xp - predicted_x),2) + np.power((test_yp - predicted_y),2))/len(test_yp))
	print('RMS Error: {}'.format(loss))
	########### Plotting ################
	plt.scatter([xp],[yp], c='y', label = 'Trained_labels')
	plt.scatter([test_xp], [test_yp], c = 'b', label = 'True Values')
	plt.scatter([predicted_x], [predicted_y], c='g', label = 'Predicted Values')
	plt.xlim([-2.5,2.5])
	plt.ylim([-2.5,2.5])
	plt.legend(loc = 'best')
	plt.show()
