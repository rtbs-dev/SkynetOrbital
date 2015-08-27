__author__ = 'Thurston Sexton, Max Yi Ren'
# This is the main entrance for SkynetOrbital
import pickle
import os.path

import simulation
import environment
import control
import design

def step(s, a):
    state_new, instant_reward = sim.step(s, a)
    return state_new, instant_reward

def act(s, d, c):
    a = c.optimal_act(s, d)
    return a

def simulate(d, c, sim):
    reward = 0.
    while sim.running():
        a = act(s, d, c)
        s, r = step(s, a)
        reward += r
    return reward, s.trajectory

def evaluate_design(d, s):
    # arg = {'control_alg': 'qlearning',
    #        'n_episode': 1000,
    #        'alpha': 1.0,
    #        'gamma': 0.9,
    #        'epsilon': 0.1,
    #        'state_space': s.state_space,
    #        'action_space': s.action_space,
    #        'sars': s,  # this can be the true simulation or the learned model
    #        'initial_state': s.initial_state,
    #        'design': d}
    arg = {'control_alg': 'bayesian',
           'n_iter': 5,
           'action_space': {'r1': [0, 12.89e3],
                            't1': [0., 400.],
                            'r2': [0, 1204.],
                            't2': [0., 630.],
                            'r3': [0, 240.]},
           'sars': s,  # this can be the true simulation or the learned model
           'initial_state': s.initial_state,
           'design': d}
    c = control.Controller(arg)

    # save controller as a pickle
    if not os.path.isfile('controller.pickle'):
        with open('controller.pickle', 'w') as f:
            pickle.dump(c.controller.control_parameter.res['max'], f)
            f.close()
    return c
    # r, t = simulate(d, c, s)
    # return r, t

d = design.Design({'mf0': 2.16e6})  # kg
sim = simulation.Simulation(d)
c = evaluate_design(design, sim)
c.controller.simulate(c.controller.control_parameter.res['max'])
