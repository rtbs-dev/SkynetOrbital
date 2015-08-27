__author__ = 'Thurston Sexton, Max Yi Ren'
# This file defines the underlying physics

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class Simulation:

    def __init__(self, design):
        # self.mf0 = input('How much fuel (kg) to bring along (can change later)?')
        # print 'setting burn scheme (call change_design() to change)'
        # self.burn = b  # a piecewise func determining throttle @ some time t
        self.design = design

        # design parameters
        self.G = 6.672e-11
        # self.M = 5.2915793e22  # Mass of Kerbin
        self.M = 5.97219e24 # Mass of Earth
        # self.R = 6e5  # Radius of Kerbin
        self.R= 6.378e6  # mean Radius of Earth
        self.F = 189e5  # Thrust of the Engine, in N
        self.m0 = 13e4  # Weight of empty rocket, in kg
        self.mf = self.design.mf0  # initial fuel mass
        # self.V0 = 6.1e2  # initial speed
        self.V0 = 50
        self.gam0 = np.pi/2.-0.1  # initial angle
        self.r_0 = self.R+2.86875e2  # initial height
        # self.phi0 = -1.5  # ???
        self.time_interval = 1.0  # second
        self.Isp = 304  # sec ???

        self.state_space = np.array([[0, 11.2e3],
                                     [-np.pi/2., np.pi/2],
                                     [self.R, 42.2e6],
                                     [0., self.design.mf0],
                                     [0]])  # speed, gamma, height, fuel mass, phi
        self.action_space = np.array([[0, 13e3]])  # burn rate, needs to be discrete for now
        self.initial_state = np.array([self.V0, self.gam0, self.r_0, self.design.mf0, 0])
        self.current_state = self.initial_state
        self.current_altitude = self.current_state[2]
        self.trajectory = np.empty((0, self.initial_state.size))
        self.action = 0

    # refresh
    def refresh(self):
        self.current_state = self.initial_state
        self.current_altitude = self.current_state[2]
        self.trajectory = np.empty((0, self.initial_state.size))
        self.action = 0


    # change design
    def change_design(self, design):
        self.design = design
        # self.design.burn = design.b  # a piecewise func determining throttle @ some time t

    # check if should stop
    def running(self, state):
        # if fuel and no crash, continue
        if state[2] >= self.current_altitude:
            return True
        else:
            # set to initial condition
            self.current_state = self.initial_state
            self.current_altitude = self.current_state[2]
            return False


    # thrust:weight ratio, based on S-IC
    # def T_w(self, g):
    #         if self.mf > self.action*self.time_interval: # if enough fuel to burn
    #             m_t = self.m0 + self.mf
    #             return self.action*g*self.Isp/(m_t*g)  # ???
    #         else:
    #             return 0.

    def T_w(self, g): #thrust:weight ratio

        #assert mf0<=2.16e6, 'Your tank is overfull (2.16e6kg max)'

        if (self.mf>=456.1e3): # S-IC
            self.Isp = 263.
            self.m0 = 130e3
            self.action_space = np.array([[0, 12.89e3]])


        elif (self.mf>=109.5e3): # S-II
            self.Isp = 360.
            self.m0 = 40.1e3
            self.action_space = np.array([[0, 1204.]])

        elif (self.mf>60e3): # S-IVB
            # note leftover stuff...too much fuel
            self.Isp = 421.
            self.m0 = 13.5e3
            self.action_space = np.array([[0, 240.]])

        else: #out of fuel
            self.Isp = 0.
            self.m0 = 13.5e3
            self.action_space = np.array([[0., 0.]])

        #Isp = 304  # sec
        v_e = g*self.Isp
        # dV = 4550  # m/s for kerbin low orbit
        #F = 18900000. #N

        #m0 = 130000. #kg
        #mf0 = 2160000 #kg
        #mf0 = 2502017*1.33# kg tsiolovsky
        #dmf = 13000.

        F = self.action*v_e

        m_t = self.m0 + self.mf
            #dmf = 80. #kg/s
        '''
        if (m_t<=m0): # or (t>405):
                F=0.  # no more fuel
                m_t=m0
        '''
        #m_t = m0 + (mf0-dmf*t)
        #print 'total mass--',m_t
        #print dmf
        #print 'T:W ratio--',F/(m_t*g)
        return [F/(m_t*g), self.action]

    # gravity turn
    def g(self, y, t):
        V_i = y[0]
        gam_i = y[1]
        r_i = y[2]
        # phi_i = y[3]



        # assert r_i >= self.R, 'You have crashed'
        g = self.G*self.M/r_i**2
        tw = self.T_w(g)
        f_v = -g*(np.sin(gam_i) - tw[0])
        f_gam = (V_i/r_i - g/V_i)*np.cos(gam_i) # + drag/ ortho. term
        f_r = V_i*np.sin(gam_i)
        f_mf = -tw[1]
        f_phi = (V_i/r_i)*np.cos(gam_i)
        # return [f_v, f_gam, f_r, f_phi]
        return f_v, f_gam, f_r, f_mf, f_phi

    # simulate one step (self.time_interval)
    def step(self, state, action):
        self.current_state = state
        self.action = action
        self.current_altitude = state[2]

        soln = odeint(self.g, self.current_state, np.linspace(0, self.time_interval, 10))
        self.trajectory = np.concatenate((self.trajectory, soln), axis=0)
        new_state = soln[-1, :]

        v = new_state[0]
        gamma = new_state[1]
        r = new_state[2]
        mf = new_state[3]
        # reward = -(gamma/np.pi)**2 - (mf/self.design.mf0)**2 - (1 - self.G*self.M/r/v**2)**2

        reward = -1*(1 - self.G*self.M/r/((v*np.cos(gamma))**2))**2
                 # *(1.-mf/self.design.mf0)\
                 # + ((gamma/np.pi*2)**2 < 0.01)*((1 - self.G*self.M/r/v**2)**2 < 0.01)*100
        # if ((gamma/np.pi*2)**2 < 0.01)*((1 - self.G*self.M/r/v**2)**2 < 0.01):
        #     wait = 1
        return new_state, reward

    # TODO: animation?
    def plot(self, soln):

        labels = ['Velocity (m/s)',
                  r'$\gamma$ (rad)',
                  'Altitude (m)',
                  'Mass (kg)',
                  'Longitude (rad)']

        fig, ax = plt.subplots(nrows=1, ncols=soln.shape[1], figsize=(25, 5))
        for i, n in enumerate(ax.flatten()):
            n.plot(np.linspace(1, soln.shape[0], soln.shape[0]), soln[:, i])
            n.set_title(labels[i])
            n.set_xlabel('time (s)')
        ax.flatten()[2].cla()
        ax.flatten()[2].plot(np.linspace(1, soln.shape[0], soln.shape[0]), soln[:, 2]-self.R)
        ax.flatten()[2].set_title(labels[2])
        # ax.flatten()[-1].plot(t, soln[:, 2]-self.R)

        fig1 = plt.figure(figsize=(10,10))
        ax1 = plt.subplot(111, polar=True)
        ax1.plot(soln[:,4], soln[:,2], linewidth=2)
        ax1.plot(np.linspace(0,2*np.pi, 10000),self.R*np.ones(10000), linewidth=3)
        ax1.fill_betweenx(self.R*np.ones(10000), np.linspace(0,2*np.pi, 10000), color='g')
        ax1.set_ylim(0., 1e6)
        ax1.set_title('Earth Orbit profile (blue)')

        circle1 = plt.Circle((0,0),self.R,color='y')

        x = np.multiply(soln[:,2], np.cos(soln[:, 4]))
        y = np.multiply(soln[:,2], np.sin(soln[:, 4]))

        fig2 = plt.figure(figsize=(10, 10))
        ax2 = plt.subplot(111, polar=False)
        ax2.plot(x, y)
        # fig2.xlim(0e5,2e5)
        # fig2.ylim(-7.5e5, -5.5e5)
        fig2 = plt.gcf()
        fig2.set_size_inches(8, 8)
        fig2.gca().add_artist(circle1)
        plt.show()
