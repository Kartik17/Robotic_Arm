import numpy as np
import matplotlib.pyplot as plt

input = np.array([2*np.cos(0.8*np.pi),2*np.sin(0.8*np.pi)])
print("Input Coordinates:{},{}".format(np.cos(0.8*np.pi),np.sin(0.8*np.pi)))


initial_guess_theta = np.array([-0.5*np.pi,0.0])

lr =0.05
error = 1.0
epoch = 0
x_graph = []
y_graph = []

#print("CHECK")
while error > 0.00001:
	#print("CHECK")
	epoch += 1
	x_predict = np.cos(initial_guess_theta[0]) + np.cos(initial_guess_theta[0]+initial_guess_theta[1])
	y_predict = np.sin(initial_guess_theta[0]) + np.sin(initial_guess_theta[0]+initial_guess_theta[1])

	x_graph.append(x_predict)
	y_graph.append(y_predict)

	error = (input[0] - x_predict)**2 + (input[1] - y_predict)**2

	dtheta1 = -2*(input[0] - x_predict)*(-np.sin(initial_guess_theta[0]) - np.sin(initial_guess_theta[0]+initial_guess_theta[1])) - 2*(input[1] - y_predict)*(np.cos(initial_guess_theta[0]) + np.cos(initial_guess_theta[0]+initial_guess_theta[1]))
	dtheta2 = -2*(input[0] - x_predict)*(- np.sin(initial_guess_theta[0]+initial_guess_theta[1])) - 2*(input[1] - y_predict)*np.cos(initial_guess_theta[0]+initial_guess_theta[1])

	initial_guess_theta[0] = initial_guess_theta[0] - lr*dtheta1
	initial_guess_theta[1] = initial_guess_theta[1] - lr*dtheta2

	if epoch%100 == 0:
		print("Error:{}".format(error))

print(epoch)
print("Predicted Coordinates:{},{}".format(x_predict,y_predict))
print("Requied Theta:{},{}".format(initial_guess_theta[0],initial_guess_theta[1]))

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x_graph,y_graph)
plt.show()

plt.plot([0.0,np.cos(initial_guess_theta[0]),x_predict],[0.0,np.sin(initial_guess_theta[0]),y_predict])
plt.xlim([-3.0,3.0])
plt.ylim([-3.0,3.0])
plt.grid()
plt.scatter([0., input[0]],[0., input[1]], marker = 'o', color = 'r')
#plt.hlines(y=0.0, xmin=-5., xmax=5., linewidth=2.)
plt.show()
