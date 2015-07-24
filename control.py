__author__ = 'Thurston Sexton, Max Yi Ren'
import qlearning

class Controller:
    def __init__(self, arg):
        self.control_alg = arg.control_alg
        if self.control_alg == 'qlearning':
            self.controller = qlearning.QLearning(arg)
        else:
            try_other_method = 1
        self.controller.learn()
        self.design = arg.desgin # for pickle

    def optimal_act(self, state, design):
        return self.controller.e_greedy(state)