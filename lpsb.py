import numpy as np
import math
import matplotlib.pyplot as plt

def trajectory_planner(Q_matrix, time, acceleration, delta_time):
	joints = Q_matrix.shape[1]-1 #Total No of points(q)
	cartesian_points = Q_matrix.shape[0] # Total no of cartesian points 
	total_time = sum(time)

	theta_all = []
	omega_all = []

	for point in range(cartesian_points):
		
		acceleration_signed_array = []
		t_blend_array = []
		straight_vel_array = []
		t_straight_array = []

		current_point_array = np.asarray(Q_matrix[point]).reshape(-1)
		delta_position_array = [y-x for x,y in zip(current_point_array,current_point_array[1:])]
		straight_vel_mid = [round(x/y,3) for x,y in zip(delta_position_array[1:-1],time[1:-1])]

		#First Blend
		sigmoid = (delta_position_array[0])/abs(delta_position_array[0])
		acceleration_signed_array.append(sigmoid*acceleration[0])
		t_blend = time[0] - math.sqrt(time[0]**2 - 2*(delta_position_array[0])/acceleration_signed_array[0])
		if t_blend > 0:
			t_blend_array.append(t_blend)
		else:
			print(t_blend,'Time to less')
			break
		straight_vel_array.append(delta_position_array[0]/(time[0]-0.5*t_blend_array[0]))
		straight_vel_array.extend(straight_vel_mid)

		#Last Blend
		sigmoid = (-delta_position_array[-1])/abs(-delta_position_array[-1])
		acceleration_signed_last = (sigmoid*acceleration[-1])
		t_blend_last =  time[-1] - math.sqrt(time[-1]**2 + 2*(delta_position_array[-1])/acceleration_signed_last)
		straight_vel_array.append(delta_position_array[-1]/(time[0]-0.5*t_blend_last))


		#signed_acceleration
		delta_straight_vel_array = [y-x for x,y in zip(straight_vel_array,straight_vel_array[1:])]
		acceleration_signed_mid = [(x/abs(x))*y for x,y in zip(delta_straight_vel_array, acceleration[1:-1])]
		acceleration_signed_array.extend(acceleration_signed_mid)
		acceleration_signed_array.append(acceleration_signed_last)
		
		# t_blend
		t_blend_mid = [x/y for x,y in zip(delta_straight_vel_array,acceleration_signed_array[1:])]
		t_blend_array.extend(t_blend_mid)
		t_blend_array.append(t_blend_last)

		#t_straight
		t_straight_array.append(round(time[0] - t_blend_array[0] - 0.5*t_blend_array[2],3))
		t_straight_mid = [round(time[i] - 0.5*(t_blend_array[i] + t_blend_array[i+1]),3) for i in range(1,len(time)-1)]
		t_straight_array.extend(t_straight_mid)
		t_straight_array.append(time[-1] - t_blend_array[-1] - 0.5*t_blend_array[-2])
	
		theta = [current_point_array[0]]
		omega = [0.]
		time_current = 0
		epoch = 0
		interval = [0.]

		for i in range(len(t_blend_array) + len(t_straight_array)):
			k = i/2 #0,0,1,1,2,2,3
			if i%2 == 0:
				interval.append(round(t_blend_array[k],1) + interval[-1])
				print(interval[-2], interval[-1])
				
				while time_current >= interval[-2] and time_current < interval[-1]:
					time_current += delta_time
					omega.append(omega[-1] + acceleration_signed_array[k]*delta_time)
					theta.append(theta[-1] + omega[-1]*delta_time + 0.5*acceleration_signed_array[k]*(delta_time**2))

			else:
				interval.append(interval[-1] + round(t_straight_array[k],1))
				while time_current >= interval[-2] and time_current < interval[-1]:
					time_current += delta_time
					omega.append(straight_vel_array[k])
					theta.append(theta[-1] + straight_vel_array[k]*delta_time)


		plt.plot(np.array(range(len(theta)))*delta_time,omega)
		plt.show()

		plt.plot(np.array(range(len(theta)))*delta_time,theta)
		#plt.plot(np.array(time),current_point_array, c = 'r', marker = '*')
		plt.show()


		theta_all.append(theta)
		omega_all.append(omega)


	return theta_all,omega_all


if __name__ == '__main__':
	Q_matrix = np.matrix([np.array([0.1,0.2,1.0]), np.array([0.1,0.2,1.0]), np.array([0.1,0.2,1.0])])
	time = np.array([10,10])
	acceleration = np.array([0.01,0.02,0.03])

	trajectory_planner(Q_matrix,time,acceleration,0.01)