__author__ = 'Thurston Sexton, Max Yi Ren'
# This is the main entrance for SkynetOrbital
import simulation
import environment
import control
import design

def step(state, action, design):
    state_new, instant_reward = simulation.step(state, action, design)
    return state_new, instant_reward

def act(state, design, controller):
    action = controller.optimal_act(state, design)
    return action

def simulate(design, controller):
    s = simulation(0)
    reward = 0.
    while s.running:
        a = act(s,design,controller)
        s, r = step(s,a,design)
        R += r
    return reward, s.trajectory

def evaluate_design(design):
    c = control.QController()
    c = c.learn(design, environment)
    r, t = simulate(design, c)
    return r, t

evaluate_design(design)