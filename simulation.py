__author__ = 'Thurston Sexton, Max Yi Ren'
# This file defines the underlying physics for project SkynetOrbital

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

R_earth = 6.371e6 # Earth Mean rad, in m
g_0 = 9.80665 # earth standard gravity, in m/s**2

def alt(x,y): #altitude above sea level
    return np.sqrt(x**2+y**2) - R_earth

def lon(x,y): # 2D from north pole down, @equator
    return np.arctan2(y,x)

def grav(x,y):
    assert alt(x,y) >= 0, 'must be @ or above sea level'
    h = alt(x,y)
    phi=lon(x,y)
    #print phi
    g_h = g_0*(R_earth/(R_earth+h))**2

    return np.array([g_h*np.cos(phi), g_h*np.sin(phi)])


mag = lambda x: np.sqrt(x.dot(x))
print grav(0., R_earth)

def T_w(t,g): #thrust:weight ratio, based on S-IC

        F = 34020000./1.8 #N

        m0 = 130000. #kg
        mf0 = 2290000. - m0 #kg

        dmf = 80. #kg/s

        m_t = m0 + (mf0-dmf*t)

        if (m_t<=m0) or (t>405):
            F=0.
            #print 'Fuel is gone'
        #print F/(m_t*g)
        return F/(m_t*g)

G = 6.672e-11 # Gravitational Constant
#M = 5.97219e24 # Mass of Earth
M = 5.2915793e22 #Mass of Kerbin
R = 6e5 #Radius of Kerbin
#R= 6.378e6 # mean Radius of Earth

def g(y,t):

    V_i = y[0]
    gam_i=y[1]
    r_i = y[2]
    phi_i = y[3]

    assert r_i>=R, 'You have crashed'
    g = G*M/r_i**2


    f_v = -g*(np.sin(gam_i) - T_w(t,g))
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

soln=odeint(g, g0, t) #[]

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