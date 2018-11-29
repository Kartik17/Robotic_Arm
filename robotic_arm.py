import numpy as np
from sympy import *
from numpy import linalg as LA
from numpy.linalg import inv
import time
import logging
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.pyplot as plt
from matplotlib import animation
import math

class robotic_arm():

	def __init__(self):
		self.joints = 0
		self.alpha = None
		self.a = None
		self.q =None
		self.d = None
		self.dh_params = {}
		self.tf_matrices_list = []

		self.current_pos = []


	def set_joints(self,joint_number):
		if joint_number > 0:
			self.joints = int(joint_number)
			self.set_dh_params()
		else:
			raise('Joints Number has to be positive')

	def set_dh_params(self):
		self.alpha = symbols('alpha0:' + str(self.joints))
		self.a = symbols('a0:' + str(self.joints))
		self.q = symbols('q1:' + str(self.joints + 1))
		self. d = symbols('d1:' + str(self.joints + 1))

	def show_dh_params(self):
		print('DH Parameters are: {}'.format(self.dh_params))

	def set_dh_param_dict(self,dh_params_values):

		for i in range(len(dh_params_values)):
			self.dh_params[self.alpha[i]] = dh_params_values[i][0]
			self.dh_params[self.a[i]] = dh_params_values[i][1]
			
			if dh_params_values[i][2] == 'D':
				self.dh_params[self.d[i]] = self.d[i]

			else:
				self.dh_params[self.d[i]] = dh_params_values[i][2]
			
			if dh_params_values[i][3] == 'R':
				self.dh_params[self.q[i]] = self.q[i]

			else:
				self.dh_params[self.q[i]] = dh_params_values[i][3]

		self.set_tranform_matrices()
			
	def set_tranform_matrices(self):
		T = eye(self.joints)
		for i in range(self.joints):
			T = T*self.TF_matrix(self.alpha[i],self.a[i],self.d[i],self.q[i]).subs(self.dh_params)
			self.tf_matrices_list.append(T)


	def show_transform_matrices(self):
		print('Transform Matrices are: {}'.format(self.tf_matrices_list)) 	
		
	@staticmethod
	def TF_matrix(alpha,a,d,q):
	    TF = Matrix([[cos(q),-sin(q), 0., a],
	                [sin(q)*cos(alpha), cos(q)*cos(alpha), -sin(alpha), -sin(alpha)*d],
	                [sin(q)*sin(alpha), cos(q)*sin(alpha),  cos(alpha),  cos(alpha)*d],
	                [   0.,  0.,  0.,  1.]])
	    return TF


	def forward_kinematics(self,theta_list):
		theta_dict = {}
		T_0G = self.tf_matrices_list[-1]
		
		for i in range(len(theta_list)):
			theta_dict[self.q[i]] = theta_list[i]

		temp = T_0G.evalf(subs = theta_dict,chop = True, maxn = 4)

		x = [np.array(temp[0,-1]).astype(np.float64)]
		y = [np.array(temp[1,-1]).astype(np.float64)]
		z = [np.array(temp[2,-1]).astype(np.float64)]

		self.current_pos.append(np.array([x,y,z]))

		return self.current_pos


	def jacobian_func(self):
		T_0G = self.tf_matrices_list[-1]
		self.jacobian_mat = [diff(T_0G[:3,-1],self.q[i]).reshape(1,3) for i in range(len(self.q))]
		self.jacobian_mat = Matrix(self.jacobian_mat).T

	def inverse_kinematics(self,guess,target):
	    error = 1.0
	    tolerance = 0.05

	    # Initial Guess - Joint Angles
	    Q = guess
	    # X,Y expression
	    # X,Y value for Target Position
	    target = np.matrix(target)
	    # Jacobian
	    self.jacobian_func()
	    #jacobian_inv = jacobian.inv("LU")
	    T_0G = self.tf_matrices_list[-1]

	    error_grad = []

	    theta_dict = {}

	    lr = 0.2
	    while error> tolerance:
	    	for i in range(len(Q)):
				theta_dict[self.q[i]] = Q[i]

	        T_q =np.matrix(self.forward_kinematics(Q)[-1])

	        delta_T = target - T_q

	        Q = Q + lr*(np.matrix(self.jacobian_mat.evalf(subs = theta_dict,chop = True, maxn = 4)).astype(np.float64).T * delta_T).T
	        Q = np.array(Q)[0]

	        prev_error = error

	        error = LA.norm(delta_T)

	        if error > 10*tolerance:
	        	lr= 0.3
	        elif error < 10*tolerance:
	        	lr = 0.2
	        error_grad.append((error-prev_error))

	        print(error)

	    return Q
	def path_plan(self,guess,target_list):
		Q_list = []
		for target in target_list:
			Q = self.inverse_kinematics(guess,target)
			predicted_coordinates = self.forward_kinematics(Q)[-1]
			print(Q)
			print('Traget: {} ,  Predicted: {}'.format(target, predicted_coordinates))
			Q_list.append(Q)
			guess = Q
		return Q_list


def init():
	line.set_data([],[])
	return line,

def animate(i):
	global transform_matrices,q,Q_list

	x = [np.array(transform_matrices[k].evalf(subs = {q[0]: Q_list[i][0], q[1]: Q_list[i][1], q[2]: Q_list[i][2], q[3]: Q_list[i][3]},chop = True, maxn = 4)[0,-1]).astype(np.float64) for k in range(len(transform_matrices))]
	y = [np.array(transform_matrices[k].evalf(subs = {q[0]: Q_list[i][0], q[1]: Q_list[i][1], q[2]: Q_list[i][2], q[3]: Q_list[i][3]},chop = True, maxn = 4)[1,-1]).astype(np.float64) for k in range(len(transform_matrices))]
	z = [np.array(transform_matrices[k].evalf(subs = {q[0]: Q_list[i][0], q[1]: Q_list[i][1], q[2]: Q_list[i][2], q[3]: Q_list[i][3]},chop = True, maxn = 4)[2,-1]).astype(np.float64) for k in range(len(transform_matrices))]
	    
	line.set_data(np.array(x),np.array(y))
	line.set_3d_properties(np.array(z))
	return line,


if __name__ == '__main__':
	arm = robotic_arm()
	arm.set_joints(4)
	arm.set_dh_param_dict([[0,0,1,'R'],[-np.pi/2.,0,0,'R'],[0,1,0,'R'],[0,1,0,'R']])
	
	transform_matrices = arm.tf_matrices_list
	q = arm.q
	target_list = [[[1.6],[1.0],[0.6]],[[1.4],[1.0],[0.4]],[[1.2],[1.0],[0.2]],[[1.0],[1.0],[0.0]],[[1.0],[0.8],[0.0]]]
	Q_list = arm.path_plan([0.,0.,0.1,0.],target_list)
	
	

	fig = plt.figure()
	ax = p3.Axes3D(fig)
	for i in range(len(target_list)):
		ax.scatter(target_list[i][0], target_list[i][1], target_list[i][2], c='r', marker='*')
	ax.set_xlim3d([0.0, 2.0])
	ax.set_xlabel('X')
	ax.set_ylim3d([0.0, 2.0])
	ax.set_ylabel('Y')
	ax.set_zlim3d([0.0, 2.0])
	ax.set_zlabel('Z')
	ax.grid()
	    
	line, = ax.plot(np.array([0.]), np.array([0.]), np.array([0.]), lw = 2.0)
	anim = animation.FuncAnimation(fig, animate, init_func=init, frames= len(target_list), interval=20, blit=True)
	plt.show()