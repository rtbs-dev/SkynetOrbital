__author__ = 'Thurston'

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import seaborn as sns

def T_w(g, mf0): #thrust:weight ratio

        Isp = 304  # sec
        v_e = g*Isp
        # dV = 4550  # m/s for kerbin low orbit

        #F = 18900000. #N

        m0 = 130000. #kg
        #mf0 = 2160000 #kg
        #mf0 = 2502017*1.33# kg tsiolovsky
        dmf = 13000.

        F = dmf*v_e

        m_t = m0 + mf0
        #dmf = 80. #kg/s

        if (m_t<=m0): # or (t>405):
            F=0.
            m_t=m0

        #m_t = m0 + (mf0-dmf*t)
        print 'total mass--',m_t

        print 'T:W ratio--',F/(m_t*g)
        return F/(m_t*g)


G = 6.672e-11 # Gravitational Constant
M = 5.97219e24 # Mass of Earth
#M = 5.2915793e22 #Mass of Kerbin
#R = 6e5 #Radius of Kerbin
R= 6.378e6 # mean Radius of Earth

def g(y,t):

    V_i = y[0]
    gam_i=y[1]
    r_i = y[2]
    phi_i = y[3]
    mf0 = y[4]
    #assert r_i>=R, 'You have crashed'
    g = G*M/r_i**2

    tw=T_w(g, mf0)

    f_v = -g*(np.sin(gam_i) - tw)
    f_gam=(V_i/r_i - g/V_i)*np.cos(gam_i) # + drag/ ortho. term
    f_r = V_i*np.sin(gam_i)
    f_phi = (V_i/r_i)*np.cos(gam_i)
    f_m = -13000.
    print 'time--',t
    #print gam_i
    return [f_v, f_gam, f_r, f_phi, f_m]

V0=  50. #6.1e2
gam0=np.pi/2.-0.1
r_0 = R+ 2.86875e2
phi0 = -1.5
#mf0 = 2502017*1.33
#g0=[V0, gam0, r_0, phi0, mf0]
t=np.linspace(0,5000,5000)

trials=0
for mf0 in np.linspace(2000000, 2.5e6, 100000):
    #V0=  50. #6.1e2
    #gam0=np.pi/2.-0.1
    #r_0 = R+ 2.86875e2
    #phi0 = -1.5
    #mf0 = 2502017*1.33
    g0=[V0, gam0, r_0, phi0, mf0]
    #t=np.linspace(0,5000,10000)
    soln=odeint(g, g0, t) #[]
    trials+=1
    print np.where(soln[:,2]<R)
    if np.where(soln[:,2]<R)[0].size==0:
        print 'True'
        break
print 'This took ', trials," trials."

fig,ax = plt.subplots(nrows=1, ncols=4, figsize=(16,4))
for i,n in enumerate(ax.flatten()[:2]):
    n.plot(t,soln[:,i])
ax.flatten()[-2].plot(t,soln[:,2]-R)
ax.flatten()[-1].plot(t,soln[:,4])
plt.show()
#

fig2=plt.figure(figsize=(10,10))
ax1 = plt.subplot(111, polar=True)
ax1.plot(soln[:,3], soln[:,2], linewidth=2)
ax1.plot(np.linspace(0,2*np.pi, 10000),R*np.ones(10000), linewidth=3)
ax1.fill_betweenx(R*np.ones(10000), np.linspace(0,2*np.pi, 10000), color='g')
ax1.set_ylim(1e4,2e7)
plt.show()