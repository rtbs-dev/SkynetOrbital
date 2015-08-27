__author__ = 'Thurston Sexton, Max Yi Ren'
import numpy as np

from bayesian_optimization.bayesian_optimization import BayesianOptimization
import simulation
import design

class DirectPolicySearch:
    def __init__(self, arg):
        self.n_iter = arg['n_iter'] # number of iterations
        self.sars = arg['sars']  # environment model (state, action) -> state
        self.action_space = arg['action_space']
        self.initial_state = arg['initial_state']
        self.current_state = self.initial_state
        self.n_state = self.current_state.shape[0]
        self.control_parameter = BayesianOptimization(self.obj, self.action_space)

    # predefined action definition
    def get_action(self, T, par):
        # action definition
        if T <= par[1]:
            action = par[0]
        elif T <= par[3]:
            action = par[2]
        else:
            action = par[4]
        return action

    # the objective function, tune reward in simulation.py
    def obj(self, r1, t1, r2, t2, r3):
        par = [r1, t1, r2, t2, r3]
        T = 0  # time counter
        reward = -1e12

        self.sars.refresh()

        # if still going up
        while self.sars.running(self.current_state):
            action = self.get_action(T, par)
            if self.current_state[3] <= 0:
                action = 0.0
            self.current_state, reward = self.sars.step(self.current_state, action)
            T += 1

        if T == 0:
            stop = 1

        if reward > -0.01:
            stop = 1

        return reward

    # check if optimization is successful
    def simulate(self, raw_par):

        t1 = raw_par['max_params']['t1']
        t2 = raw_par['max_params']['t2']
        r1 = raw_par['max_params']['r1']
        r2 = raw_par['max_params']['r2']
        r3 = raw_par['max_params']['r3']
        par = [r1,t1,r2,t2,r3]

        self.current_state = self.initial_state
        sars_set = []

        T = 0  # time counter
        reward = 0
        # if still going up
        while self.sars.running(self.current_state):
            action = self.get_action(T, par)
            if self.current_state[3] <= 0:
                action = 0.0
            s, reward = self.sars.step(self.current_state, action)
            sars_set.append(np.concatenate((self.current_state, [action], [reward], s), axis=0))
            self.current_state = s
            T += 1
        self.sars.plot(np.array(sars_set)[:, 0:self.n_state])

    # main function
    def learn(self):
        print self.n_iter
        self.control_parameter.maximize(n_iter = self.n_iter)

