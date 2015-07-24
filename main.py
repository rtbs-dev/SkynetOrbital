__author__ = 'Thurston Sexton, Max Yi Ren'
# This is the main entrance for SkynetOrbital
import simulation
import environment
import control
import design

def step(s, a, d):
    state_new, instant_reward = sim.step(s, a, d)
    return state_new, instant_reward

def act(s, d, c):
    a = c.optimal_act(s, d)
    return a

def simulate(d, c, s):
    reward = 0.
    while s.running:
        a = act(s, d, c)
        s, r = step(s, a, d)
        reward += r
    return reward, s.trajectory

def evaluate_design(d, s):
    arg = {'control_alg': 'qlearning',
           'n_episode': 100,
           'alpha': 0.3,
           'gamma': 0.9,
           'eps': 0.1,
           'state_space': s.state_space,
           'action_space': s.action_space,
           'sars': s,  # this can be the true simulation or the learned model
           'initial_state': s.initial_state,
           'design': d}
    c = control.Controller(arg)
    r, t = simulate(d, c, s)
    return r, t

d = design.Design({'mf0',10000,'burn',1})
sim = simulation.Simulation(d)
evaluate_design(design, sim)