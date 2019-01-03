# Libraries
import numpy as np
import matplotlib.pyplot as plt
from pid_controller import PIDController
from lpsb import trajectory_planner
# DC Motor class

class Dc_motor():
    def __init__(self):
        self.V = 24.
        self.R = 10.
        self.L = 0.24
        self.b = 0.000001
        self.kb = 0.02
        self.kt = self.kb
        self.I_motor = 10.0
        self.I_load = 0.0
        self.delta_t = 0.01
        self.load_torque = 0.0
        self.N = 1.0
        self.n = 1.0

        self.i_phase = [0.]
        self.ang_speed = [0.]
        self.theta = [0.]
        self.theta_output = [0.]

        self.control = False
        self.theta_desired = 0.
    
    def set_motor_param(self,R,L,b,kb,kt,I_motor):
        self.R, self.L, self.b, self.kb, self.kt, self.I_motor = R,L,b,kb,kt, I_motor

    def set_load_inertia(self,load):
        self.I_load = load

    def set_sample_time(self,delta_t):
        self.delta_t = delta_t
    
    def set_load_torque(self,load_torque):
        self.load_torque = load_torque

    def set_gearbox_param(self,n, effi):
        self.N = n
        self.n = effi

    def set_voltage(self,V):
        self.V = V

    def check_params(self):
        print('No Load Response: {}'.format(self.V/self.kb))
        print('Stall Torque: {}'.format(self.kt*(self.V/self.R)))
        print('Time Constant: {:2f}'.format(self.L/self.R))

    def set_duration(self,duration):
        self.duration = duration

    def set_controller(self):
        self.control = True
    
    def motor_path(self,omega_desired):
        self.omega_desired = omega_desired

    def update(self,pid):
        time_steps = int(self.duration/self.delta_t)
        print(time_steps)
        for i in xrange(1,time_steps):
        # Current
            pid.setTarget(self.omega_desired[i])
            self.i_phase.append((self.delta_t/self.L)*(self.V - self.i_phase[i-1]*self.R - self.kb*self.ang_speed[i-1]) + self.i_phase[i-1])
            if self.i_phase[i]>= (self.V/self.R):
                self.i_phase[i] = (self.V/self.R)
            elif self.i_phase[i]<=0:
                self.i_phase[i] = 0.

            # Angular Speed
            self.ang_speed.append((self.delta_t/(self.I_motor + self.I_load/(self.N**2))*(self.kt*self.i_phase[i-1] - self.load_torque/(self.n*self.N) - self.b*self.ang_speed[i-1]) + self.ang_speed[i-1]))
            if self.ang_speed[i] <=0:
                self.ang_speed[i] = 0.
            elif self.ang_speed[i] >= (self.V/self.kb):
                self.ang_speed[i] = (self.V/self.kb)
        
            self.theta.append((self.delta_t)*self.ang_speed[i-1] +self.theta[i-1])
            self.theta_output.append(self.theta[i]*self.N)

            time = delta_t*i
            
            #if time > 10.0:
                #pid.setTarget(400)
            if self.control:
                #self.V = pid.update(self.ang_speed[i],time)
                self.V = pid.update(self.ang_speed[i],time)
    
    def plot_graph(self):
        fig,ax = plt.subplots(2,2)
        time = delta_t* np.array(range(len(self.i_phase)))
        ax[0,0].plot(time, self.i_phase)
        ax[1,0].plot(time, self.ang_speed)
        ax[0,1].plot(time, self.theta_output)
        ax[1,1].plot(self.ang_speed, self.kt*np.array(self.i_phase))

        plt.show()

if __name__ == '__main__':
    # Parameters
    V = 24
    R = 10
    L = 0.24
    b = 0.000001
    kb = 0.02
    kt = 0.02
    I_motor = 9e-6
    I_load = 0.
    N = 1
    n = 1
    delta_t = 0.001
    T_load = 0.01

    motor_1 = Dc_motor()
    motor_1.set_motor_param(R,L,b,kb,kt,I_motor)
    motor_1.set_load_torque(T_load)
    motor_1.set_voltage(V)
    motor_1.set_duration(30.0)
    motor_1.set_controller()
    motor_1.set_sample_time(0.001)

    pid = PIDController(kp = 0.6, ki = 3, kd = 0.001, max_windup = 20,
            start_time = 0, alpha = 0.75, u_bounds = [0.0, V])
    Q_matrix = np.matrix(np.array([10,1000,3000,5000]))
    time = np.array([10,10,10])
    acceleration = np.array([40,35,60,50])

    theta_desired,omega_desired = trajectory_planner(Q_matrix,time,acceleration,delta_t)
    motor_1.motor_path(omega_desired)
    print(len(omega_desired))

    #pid.setTarget(800)
    motor_1.update(pid)
    motor_1.plot_graph()