import numpy as np
from sympy import *
from numpy import linalg as LA
from numpy.linalg import inv



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

def jacobian_func(T,q):
    jacobian_mat = []
    for i in range(len(q)):
        jacobian_mat.append(diff(T[:2,-1],q[i]).reshape(1,len(q)))

    jacobian_mat = Matrix(jacobian_mat).T
    return jacobian_mat

T_01 = TF_matrix(alpha0,a0,d1,q1).subs(subs_dict)
T_12 = TF_matrix(alpha1,a1,d2,q2).subs(subs_dict)
T_23 = TF_matrix(alpha2,a2,d3,q3).subs(subs_dict)
T_03 = T_01*T_12*T_23

error = 1.0
tolerance = 0.01

Q = np.matrix([[np.pi/3.],[np.pi/3.]])

gen_gripper = T_03[:2,-1]

target = np.matrix([[0.],[2.0]])


jacobian = jacobian_func(T_03,[q1,q2])
jacobian_inv = jacobian.inv("LU")

while error> tolerance:
    #print('Check')
    T_q = np.matrix(gen_gripper.evalf(subs = {q1: Q[0,0], q2: Q[1,0], q3: 0.})).astype(np.float64)

    delta_T = target - T_q

    Q = Q + np.matrix(jacobian_inv.evalf(subs = {q1: Q[0,0], q2: Q[1,0], q3: 0.})).astype(np.float64) * delta_T

    error = LA.norm(delta_T)

    print(error)
print(Q)
print(np.matrix(T_03.evalf(subs = {q1: Q[0,0], q2: Q[1,0], q3: 0.})[:2,-1]))
