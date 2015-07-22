__author__ = 'Thurston Sexton, Max Yi Ren'
import pickle
import os.path
import numpy as np
from matplotlib import pyplot as plt
import sklearn

'''
This file specifies the environment class which
either learns a model from data.pickle or directly
loads a model if model.pickle exists.

* DATA ENCODING
x: independent variable
input_state_id: state indices
input_action_id: action indices
input_design_id: design indices (optional?)

y: dependent variable(new action and reward)
output_action_id: action indices
output_reward_id: reward indices
'''
class Environment:
    def __init__(self):
        if os.path.isfile('model.pickle'):
            with open('model.pickle') as f:
                self.environment_model, self.reward_model = pickle.load(f)
                f.close()
        elif os.path.isfile('data.pickle'):
            with open('data.pickle') as f:
                self.data = pickle.load('data.pickle')
                f.close()
            self.learn()
            with open('model.pickle', 'w') as f:
                pickle.dump([self.environment_model, self.reward_model], f)
                f.close()

    def learn(self):
        X = self.data.X
        y = self.data.y
        input_state_id = self.data.input_state_id
        input_action_id = self.data.input_action_id
        input_design_id = self.data.input_design_id
        output_action_id = self.data.output_action_id
        output_reward_id = self.data.output_reward_id

        # TODO: implement training-test (or crossvalidation) for model selection
        # TODO: implement general routine of crossvalidation for model parameter selection

        ## train environment model
        clf = sklearn.linear_model.LinearRegression()
        clf.fit(X,y[:,output_action_id])
        self.environment_model = clf

        ## train reward model
        clf = sklearn.linear_model.LinearRegression()
        clf.fit(X,y[:,output_reward_id])
        self.reward_model = clf

    def predict(self, state, action, design):
        p = self.input_state_id.size + self.input_action_id.size + self.input_design_id.size
        x = np.zeros((p,))
        x[self.input_state_id] = state
        x[self.input_action_id] = action
        x[self.input_design_id] = design
        new_action = self.environment_model.predict(x)
        new_reward = self.reward_model.predict(x)
        return new_action, new_reward

    def step(self, state, action, design):
        new_state, reward = self.predict(state, action, design)
        return new_state, reward

