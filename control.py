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
        # self.controller.obj(30924.4887,58.2323,36322.4821,49.0338,29128.3738)
        # self.controller.simulate({'max_params':{'r1':30924.4887,'t1':58.2323,'r2':36322.4821,'t2':49.0338,'r3':29128.3738}})
        #
        self.controller.learn()
        self.controller.simulate(self.controller.control_parameter.res['max'])

    def optimal_act(self, state, design):
        if self.control_alg == 'qlearning':
            return self.controller.e_greedy(state)
