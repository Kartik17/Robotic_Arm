import numpy as np
from sympy import *
import matplotlib.pyplot as plt

alpha0,alpha1,alpha2,alpha3,alpha4,alpha5,alpha6 = symbols('alpha0:7')
a0,a1,a2,a3,a4,a5,a6 = symbols('a0:7')
d1,d2,d3,d4,d5,d6,d7 = symbols('d1:8')
q1,q2,q3,q4,q5,q6,q7 = symbols('q1:8')

# Create Modified DH parameters
subs_dict = {   alpha0:0 , a0:0, d1: 0,q1: q1,
                alpha1:0 , a1:10, d2:0, q2: q2,
				alpha2:0 , a2:10, d3:0, q3: q3}

# Define Modified DH Transformation matrix

def TF_matrix(alpha,a,d,q):
        TF = Matrix([[cos(q),-sin(q), 0, a],
                    [sin(q)*cos(alpha), cos(q)*cos(alpha), -sin(alpha), -sin(alpha)*d],
                    [sin(q)*sin(alpha), cos(q)*sin(alpha), cos(alpha),  cos(alpha)*d],
                    [   0,  0,  0,  1]])
        return TF

T_01 = TF_matrix(alpha0,a0,d1,q1).subs(subs_dict)
T_12 = TF_matrix(alpha1,a1,d2,q2).subs(subs_dict)
T_23 = TF_matrix(alpha2,a2,d3,q3).subs(subs_dict)

T_03 = T_01*T_12*T_23

trial = np.matrix(T_03.evalf(subs = {q1:0,q2:0,q3:0}))
#r_vec = np.matrix([[0],[0],[0],[1]])


q1_list = np.linspace(-1,1,10)
q2_list = np.linspace(-1,1,5)

x_pos =[]
y_pos =[]

for i in range(len(q1_list)):
    for  j in range(len(q2_list)):
        r_vec = np.matrix(T_03.evalf(subs = {q1:q1_list[i],q2:q2_list[j],q3:0}))*np.matrix([[0],[0],[0],[1]])
        output_theta.append([q1_list[i],q2_list[j]])
        x_pos.append(float(r_vec[0][0]))
        y_pos.append(float(r_vec[1][0]))

plt.scatter(x_pos[:],y_pos[:])
plt.show()
