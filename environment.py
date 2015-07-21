__author__ = 'Thurston Sexton, Max Yi Ren'
import pickle

class Environment:
    def __init__(self):
        self.model = pickle.load('model.pickle')

    def step(self, state, action, design):
        new_state, reward = self.model(state, action, design)
        return new_state, reward
