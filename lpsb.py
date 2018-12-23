import numpy as np
import matplotlib.pyplot as plt

def joint_trajectory(const_Vel, q_init, q_final, total_time):
	if const_Vel <= 2*(q_final - q_init)/total_time & const_Vel > (q_final - q_init)/total_time
		pass
	else:
		print('Motion not possible')
		exit()

	t_blend = (q_final - q_init + const_Vel*total_time)/const_Vel
	a = const_Vel/t_blend

	return t_blend, a

def update_joint_position(t_init, t_final, sampling_time):
	q = [0,1,2]
	time_q = [1,1]
	vel_q = [40,40]

	if time_q.sum() == (t_final - t_init):
		print('Perfect')
		pass
	else:
		print('Your watch is broken, thou shall not pass!')

	for i in range(len(q) - 1):
		q_init = q[i]
		q_final = q[i+1]
		total_time = t_init + time_q[i]
		const_Vel = vel_q[i]

		t_blend , a = joint_trajectory(const_Vel, q_init, q_final, t_init)

		for t in np.linspace(t_init, t_final, sampling_time):
			if t>=0 & t<=t_blend:
				q_pos.append(q_init + 0.5*a*t*t)

			elif t > t_blend & t <= ()
