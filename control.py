__author__ = 'Thurston Sexton, Max Yi Ren'
import qlearning
import directpolicy

class Controller:
    def __init__(self, arg):
        self.control_alg = arg['control_alg']
        if self.control_alg == 'qlearning':
            self.controller = qlearning.QLearning(arg)
        elif self.control_alg == 'bayesian':
            self.controller = directpolicy.DirectPolicySearch(arg)

        # check hand-tuned solution
        # self.controller.obj(13000,500,0,0,0)
        # self.controller.simulate([13000,500,0,0,0])

        self.controller.learn()
        self.controller.simulate(self.controller.control_parameter.res['max'])

    def optimal_act(self, state, design):
        if self.control_alg == 'qlearning':
            return self.controller.e_greedy(state)
