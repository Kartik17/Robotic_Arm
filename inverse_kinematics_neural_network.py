import numpy as np
import matplotlib.pyplot as plt

def weights(mean = 0.0, std = 1.0 ,size=(0,0)):
	return np.random.normal(mean,std,size)

def sigmoid(x):
	return 1.0/(1.0+ np.exp(-x))

def relu(x):
	x[x<0] = 0
	return x

def der_relu(x):
	x[x>0] = 1
	x[x<0] = 0
	return x

def der_sigmoid(x):
	return np.multiply(x,(1.0-x))

def loss_func_mse(actual_y,predicted_y):
	return np.matrix.sum(np.power((predicted_y - actual_y),2))
	
def loss_func_mse_der(actual_y,predicted_y):
	return 2*(predicted_y-actual_y)

def loss_func_abs(actual_y,predicted_y):
	return np.sum(abs(predicted_y - actual_y))

def loss_func_abs_der(actual_y,predicted_y):
	a = predicted_y - actual_y
	a[a>0] = 1
	a[a<0] = -1
	return a

	
# Output is the Angle in radians between pi and -pi	
input_max = np.pi 
input_min = -0.99*np.pi 
data_points = 301
func = 	np.linspace(input_min,input_max,data_points)
	
actual_y = np.matrix(func).T # 4x1 [[0.0],[1.0],[4.0],[9.0],[16.0],[24.0],[36.0]]
a = (actual_y.max()-actual_y.min())
b =actual_y.min()
actual_y = (actual_y - b)/a



# Input is the x,y coordinates of the arm.From pi to -pi, cos(theta) and sin(theta)
data = np.matrix([np.cos(np.linspace(input_min,input_max,data_points)),np.sin(np.linspace(input_min,input_max,data_points))]).T #601x2
mean_data = data.mean(axis = 0)
std_data = data.std(axis = 0)
data = (data - mean_data)/std_data


hidden_layers_1 = 100
hidden_layers_2 = 50
input_layers = 2
output_nodes = 1
lr = 0.005


# There are two hidden layers in the Deep network. Weights and bias of Hidden and output is initialized below.
weights_1 = np.matrix(weights(size = (input_layers,hidden_layers_1))) #2x40
bias_1 = np.matrix(weights(size = (1,hidden_layers_1))) #1x40
weights_2 = np.matrix(weights(size = (hidden_layers_1,hidden_layers_2))) #40x20
bias_2 = np.matrix(weights(size = (1,hidden_layers_2))) #1x20
weights_3 = np.matrix(weights(size = (hidden_layers_2,output_nodes))) #20x1
bias_3 = np.matrix(weights(size = (1,output_nodes))) #1x1


loss_graph = []
error = 1.
epoch = 0
threshold = 0.0000001

while error > threshold and epoch<100000:
	epoch +=1
	logits_1 = np.add(np.matmul(data,weights_1),bias_1) # 601x2 2x40 : 601x40
	activated_layer_1 = sigmoid(logits_1) # 21x15
	logits_2 = np.add(np.matmul(activated_layer_1,weights_2),bias_2) # 21x15 15x6 : 21x6
	activated_layer_2 = sigmoid(logits_2) # 21x6
	logits_3 = np.add(np.matmul(activated_layer_2,weights_3),bias_3) # 21x6 6x1 : 21x1
	predicted_y = sigmoid(logits_3) # 21x1

	dweights_3 = np.dot(activated_layer_2.T,np.multiply(loss_func_mse_der(actual_y,predicted_y),der_sigmoid(predicted_y))) # 15x21,21x1: 15x1
	dweights_2 = np.dot(activated_layer_1.T,np.multiply((np.dot(np.multiply(loss_func_mse_der(actual_y,predicted_y),der_sigmoid(predicted_y)),weights_3.T)),der_sigmoid(activated_layer_2)))#1x21,(21x1,1x15) : 1x15
	dweights_1 = np.dot(data.T,np.multiply(np.dot(np.multiply((np.dot(np.multiply(loss_func_mse_der(actual_y,predicted_y),der_sigmoid(predicted_y)),weights_3.T)),der_sigmoid(activated_layer_2)),weights_2.T),der_sigmoid(activated_layer_1)))#1x21,(21x1,1x15) : 1x15
	
	
	dbias_3 = np.multiply(loss_func_mse_der(actual_y,predicted_y),der_sigmoid(predicted_y)).sum(axis=0) #21x1: 1x1
	dbias_2 = np.multiply((np.dot(np.multiply(loss_func_mse_der(actual_y,predicted_y),der_sigmoid(predicted_y)),weights_3.T)),der_sigmoid(activated_layer_2)).sum(axis=0)#21x1,1x15: 21x15: 1x15
	dbias_1 = np.multiply(np.dot(np.multiply(np.dot(np.multiply(loss_func_mse_der(actual_y,predicted_y),der_sigmoid(predicted_y)),weights_3.T),der_sigmoid(activated_layer_2)),weights_2.T),der_sigmoid(activated_layer_1)).sum(axis=0)#21x1,1x15: 21x15: 1x15

	weights_1 = weights_1 -lr*dweights_1
	weights_2 = weights_2 -lr*dweights_2
	weights_3 = weights_3 -lr*dweights_3
	
	bias_1 = bias_1 - lr*dbias_1
	bias_2 = bias_2 - lr*dbias_2
	bias_3 = bias_3 - lr*dbias_3
	
	error = loss_func_mse(actual_y,predicted_y)

	if epoch%10000 == 0:
		loss_graph.append(error)
		print('Error at {}:{}'.format(epoch,error))

print('Epochs:{}'.format(epoch))
#print('Predicted y: {}'.format(((predicted_y*2)-1)))
print('Weights:{}'.format(weights_1))
print('Bias:{}'.format(bias_1))


plt.plot(np.linspace(input_min,input_max,data_points),np.cos(np.array(predicted_y[:,0]*a+b)), c ='r', label = 'Predicted')
plt.plot(np.linspace(input_min,input_max,data_points),np.cos(func), label = 'Actual')
plt.legend(loc = 'best')
plt.show()

# Loss graph per 10000 epoch
fig = plt.figure()
ax_1 = fig.add_subplot(211)
ax_1.plot(range(0,len(loss_graph)),loss_graph)


# Should Plot a Circle
ax_2 = fig.add_subplot(212)
ax_2.plot(np.cos(np.array(predicted_y[:,0]*a+b)),np.sin(np.array(predicted_y[:,0]*a+b)))

plt.show()

'''
# To test new data.

test_data = np.matrix(np.linspace(input_min,input_max,data_points)).T # 7x1
test_data_norm = (test_data - mean_data)/std_data

logits_11 = np.add(np.matmul(test_data_norm,weights_1),bias_1) # 7x1 1x15 : 7x15
activated_layer_11 = sigmoid(logits_11) # 7x15
logits_22 = np.add(np.matmul(activated_layer_11,weights_2),bias_2) # 7x15 15x1 : 7x1
activated_layer_22 = sigmoid(logits_22) # 7x1
logits_33 = np.add(np.matmul(activated_layer_22,weights_3),bias_3) # 7x1 1x15 : 7x15
predicted_test = sigmoid(logits_33) # 7x15

plt.plot(np.linspace(input_min,input_max,data_points),(np.array(predicted_test[:,0]*a)+b), label = 'Actual')
plt.show()'''
#print(np.cos(test_data))
