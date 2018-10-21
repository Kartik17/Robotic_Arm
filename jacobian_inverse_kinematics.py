import numpy as np
from sympy import *
from numpy import linalg as LA
from numpy.linalg import inv
import time
import logging

#################### Logging Function: Decorator ###################

def note_time_func(use_func):
    def wrapper_func(*args,**kwargs):
        time_start = time.time()
        result = use_func(*args,**kwargs)
        delta_time = time.time() - time_start
        print('Time to run the function - {} -  is: {}'.format(use_func.__name__,delta_time))
        return result

    return wrapper_func

######################## Forward Kinematics ########################

# Create Symbolic variables
alpha0,alpha1,alpha2,alpha3,alpha4,alpha5,alpha6 = symbols('alpha0:7')
a0,a1,a2,a3,a4,a5,a6 = symbols('a0:7')
d1,d2,d3,d4,d5,d6,d7 = symbols('d1:8')
q1,q2,q3,q4,q5,q6,q7 = symbols('q1:8')

# Create Modified DH parameters, Two link PLanar arm:
subs_dict = {   alpha0:0 , a0:0, d1: 0,q1: q1,
                alpha1:0 , a1:1., d2:0, q2: q2,
				alpha2:0 , a2:1., d3:0, q3: q3}

# Define Modified DH Transformation matrix

def TF_matrix(alpha,a,d,q):
    TF = Matrix([[cos(q),-sin(q), 0, a],
                [sin(q)*cos(alpha), cos(q)*cos(alpha), -sin(alpha), -sin(alpha)*d],
                [sin(q)*sin(alpha), cos(q)*sin(alpha), cos(alpha),  cos(alpha)*d],
                [   0,  0,  0,  1]])
    return TF

######################## Inverse Kinematics ########################

@note_time_func # Equivalent to: jacobian_func = note_time_func(jacobian_func)
def jacobian_func(T,q):
    jacobian_mat = [diff(T[:2,-1],q[i]).reshape(1,len(q)) for i in range(len(q))]
    jacobian_mat = Matrix(jacobian_mat).T
    return jacobian_mat

T_01 = TF_matrix(alpha0,a0,d1,q1).subs(subs_dict)
T_12 = TF_matrix(alpha1,a1,d2,q2).subs(subs_dict)
T_23 = TF_matrix(alpha2,a2,d3,q3).subs(subs_dict)
T_02 = T_01*T_12
T_03 = T_01*T_12*T_23

# Important Note: If you skip the astype method, numpy will create a matrix of type 'object',
# which won't work with common array operations.

@note_time_func
def iteration(guess,target_list, T_03):
    error = 1.0
    tolerance = 0.01

    # Initial Guess - Joint Angles
    Q = guess
    # X,Y expression
    gen_gripper = T_03[:2,-1]
    # X,Y value for Target Position
    target = np.matrix(target_list)
    # Jacobian
    jacobian = jacobian_func(T_03,[q1,q2])
    jacobian_inv = jacobian.inv("LU")

    
    while error> tolerance:
        T_q = np.matrix(gen_gripper.evalf(subs = {q1: Q[0,0], q2: Q[1,0], q3: 0.})).astype(np.float64)

        delta_T = target - T_q

        Q = Q + np.matrix(jacobian_inv.evalf(subs = {q1: Q[0,0], q2: Q[1,0], q3: 0.})).astype(np.float64) * delta_T

        error = LA.norm(delta_T)

        print(error)

    return Q

Q_list = []
transform_matrices = [T_01,T_02,T_03]
guess = np.matrix([[0.],[np.pi/3.]])
target_list = [[[0.],[2.0]],[[0.2],[1.8]],[[0.4],[1.4]],[[0.6],[1.0]],[[0.8],[1.0]],[[1.0],[1.0]],[[1.5],[0.5]],[[2.],[0.0]]]
for target in target_list:
    Q = iteration(guess,target,T_03)
    Q_list.append(Q)
    guess = Q

#Q= iteration(T_03)
import matplotlib.pyplot as plt
from matplotlib import animation


def init():
    line.set_data([],[])
    return line,

def animate(i):
    global Q_list, transform_matrices
    x = [np.array(transform_matrices[k].evalf(subs = {q1: Q_list[i][0,0], q2: Q_list[i][1,0], q3: 0.})[0,-1]).astype(np.float64) for k in range(len(transform_matrices))]
    y = [np.array(transform_matrices[k].evalf(subs = {q1: Q_list[i][0,0], q2: Q_list[i][1,0], q3: 0.})[1,-1]).astype(np.float64) for k in range(len(transform_matrices))]

    line.set_data(x,y)
    return line,

def plot_arm(Q,transform_matrices):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    # Check For Lambdify to Convert to Numpy matrix
    x = [np.array(transform_matrices[i].evalf(subs = {q1: Q[0,0], q2: Q[1,0], q3: 0.})[0,-1]).astype(np.float64) for i in range(len(transform_matrices))]
    y = [np.array(transform_matrices[i].evalf(subs = {q1: Q[0,0], q2: Q[1,0], q3: 0.})[1,-1]).astype(np.float64) for i in range(len(transform_matrices))]
    ax1.plot(x,y)
    ax1.set_xlim(-2.5,2.5)
    ax1.set_ylim(-2.5,2.5)
    ax1.grid(linestyle ='--')
    plt.show()

#plot_arm(Q,[T_01,T_02,T_03])

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-2.5, 2.5), ylim=(-2.5, 2.5))
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.grid()
line, = ax.plot([], [], '-o', lw=2, c = 'b')

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=10, interval=1000, blit=True)
plt.show()

print(Q)
print(np.matrix(T_03.evalf(subs = {q1: Q[0,0], q2: Q[1,0], q3: 0.})[:2,-1]))
