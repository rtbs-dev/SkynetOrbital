__author__ = 'Thurston Sexton, Max Yi Ren'
# This file defines the underlying physics for project SkynetOrbital

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class Simulation:

    G = 6.672e-11 # Gravitational Constant
    #M = 5.97219e24 # Mass of Earth
    M = 5.2915793e22 #Mass of Kerbin
    R = 6e5 #Radius of Kerbin
    #R= 6.378e6 # mean Radius of Earth

    def __init__(self, b):
        self.mf0 = input('How much fuel (kg) to bring along (can change later)?')
        print 'setting burn scheme (call change_design() to change)'
        self.burn = b # a piecewise func determining throttle @ some time t

        self.G = 6.672e-11
        self.M_k = 5.2915793e22 # Mass of Kerbin
        self.R_k = 6e5 # Radius of Kerbin

    def change_design(self, mf0, b):

        self.mf0 = mf0 #new init. fuel
        self.burn = b # a piecewise func determining throttle @ some time t

    def T_w(self,t,g): #thrust:weight ratio, based on S-IC

            F = 189e5 # Thrust of the Engine, in N

            m0 = 13e4 # Weight of empty rocket, in kg

            #mf0 = 229e4 - m0 # Mass of initial fuel, in kg
            mf0 = self.mf0 #inital fuel
            #dmf = 80. #kg/s
            dmf = self.burn(t) #already burned fuel

            m_t = m0 + (mf0-dmf)

            if m_t<=m0: # or (t>405) # 405s only for original case
                F=0.
                #print 'Fuel is gone'
            #print F/(m_t*g)
            return F/(m_t*g)



    def g(self,y,t):

        V_i = y[0]
        gam_i=y[1]
        r_i = y[2]
        phi_i = y[3]

        assert r_i>=self.R, 'You have crashed'
        g = self.G*self.M/r_i**2


        f_v = -g*(np.sin(gam_i) - self.T_w(t,g))
        f_gam=(V_i/r_i - g/V_i)*np.cos(gam_i) # + drag/ ortho. term
        f_r = V_i*np.sin(gam_i)
        f_phi = (V_i/r_i)*np.cos(gam_i)
        #print t
        #print gam_i
        return [f_v, f_gam, f_r, f_phi]

    V0= 6.1e2
    gam0=1.5
    r_0 = R+2.86875e2
    phi0 = -1.5
    g0=[V0, gam0, r_0, phi0]
    t=np.linspace(0,5000,10000)

    # TODO: put state, action and design into the problem formulation
    def step(self, state, action, design):
        soln=odeint(self.g, self.g0, self.t)
        return soln

    def plot(self, soln):
        fig,ax = plt.subplots(nrows=1, ncols=3, figsize=(12,4))
        for i,n in enumerate(ax.flatten()[:2]):
            n.plot(t,soln[:,i])
        ax.flatten()[-1].plot(t,soln[:,2]-R_earth)

        fig=plt.figure(figsize=(10,10))
        ax1 = plt.subplot(111, polar=True)
        ax1.plot(soln[:,3], soln[:,2], linewidth=2)
        ax1.plot(np.linspace(0,2*np.pi, 10000),R*np.ones(10000), linewidth=3)
        ax1.fill_betweenx(R*np.ones(10000), np.linspace(0,2*np.pi, 10000), color='g')
        ax1.set_ylim(1e4,1e6)

        circle1=plt.Circle((0,0),R,color='y')

        x=np.multiply(soln[:,2], np.cos(soln[:,3]))
        y=np.multiply(soln[:,2], np.sin(soln[:,3]))
        plt.plot(x, y)
        plt.xlim(0e5,2e5)
        plt.ylim(-7.5e5, -5.5e5)
        fig = plt.gcf()
        fig.set_size_inches(8, 8)
        fig.gca().add_artist(circle1)