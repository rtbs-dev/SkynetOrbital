__author__ = 'Thurston Sexton, Max Yi Ren'
import numpy as np
import itertools
from sklearn.gaussian_process import GaussianProcess
from sklearn.svm import SVR

class QLearning:
    def __init__(self, arg):
        self.n_episode = arg['n_episode']
        self.alpha = arg['alpha']  # learning rate
        self.gamma = arg['gamma']  # depreciation of reward
        self.epsilon = arg['epsilon']
        self.state_space = arg['state_space']  # state and action space can be multidimensional
        self.action_space = arg['action_space']
        self.n_state = self.state_space.shape[0]
        self.n_action = self.action_space.shape[0]
        self.sars = arg['sars']  # environment model (state, action) -> state
        self.sars_set = []  # store visited state-action-reward-state data
        # self.predict_q = GaussianProcess(theta0=0.1, thetaL=.001, thetaU=1., regr='linear')  # isotropic gaussian
        self.predict_q = SVR(C=1.0, epsilon=0.2)
        self.initial_state = arg['initial_state']
        self.current_state = self.initial_state
        self.current_action = []
        self.current_observation = []
        self.design = arg['design']

    # pick action with highest q value based on state
    def best_action(self, s):
        feasible_action = self.get_feasible_action(s)
        Q = np.zeros((feasible_action.size,))
        for i, action in enumerate(feasible_action):
            try:
                Q[i] = self.predict_q.predict(np.concatenate((self.current_state, action)))
            except:
                Q[i] = np.random.rand() # before any training, randomly pick an action
        return max(Q), feasible_action[np.where(Q == max(Q))[0]][0]

    # get list of feasible actions from current state
    def get_feasible_action(self, state):
        mf = state[3]
        if mf >= 0:
            # TODO: need a better implementation
            return np.array(list(itertools.product(self.action_space[0]))).T[0]
        else:
            return np.array([0.])

    # train q predictor from updated saq tuple
    def train_q(self):
        current_s = np.array(self.current_observation)[:, np.arange(0, self.n_state)]
        current_a = np.array(self.current_observation)[:, np.arange(self.n_state, self.n_state+self.n_action)]
        current_r = np.array(self.current_observation)[:, self.n_state+self.n_action]
        current_ss = np.array(self.current_observation)[:, np.arange(self.n_state+self.n_action+1, self.n_state*2+self.n_action+1)]
        current_q = self.update_q(current_s, current_a, current_r, current_ss)

        try:
            self.predict_q.fit(np.concatenate((current_s, current_a), axis=1), current_q)
        except:
            wait = 1

    # update q for sa couple and current q predictor
    def update_q(self, s, a, r, ss):
        # TODO: need to initialize predict_q so that it spits random stuff out at the beginning
        try:
            q = self.predict_q.predict(np.concatenate((s, a), axis=1))  # current q
        except:
            q = np.random.rand(s.shape[0])
        qq = np.zeros((q.size,))  # best q for ss
        # get best q for next state "ss"
        for i, next_state in enumerate(ss):
            qq[i], aa = self.best_action(next_state)
        q = (1-self.alpha)*q + self.alpha*(r + self.gamma*qq)
        return q

    # epsilon-greedy search
    def e_greedy(self, s):
        p = np.random.rand()
        if p > self.epsilon:
            maxQ, a = self.best_action(s)
        else:
            feasible_action = self.get_feasible_action(s)
            a = np.random.choice(feasible_action)
        return a

    # start a new episode
    def reset(self):
        self.current_state = self.initial_state
        self.current_action = []
        self.current_observation = []

    # run an episode, currently do not update q model online
    def run(self):
        while self.sars.running(self.current_state):
            self.current_action = self.e_greedy(self.current_state)  # take action
            s, r = self.sars.step(self.current_state,self.current_action)  # get reward and new state
            self.current_observation.append(np.concatenate((self.current_state, [self.current_action], [r], s), axis=0))
            self.current_state = s

    # main function
    def learn(self):
        for e in np.arange(self.n_episode):
            self.run()
            self.sars_set.append(self.current_observation)
            if np.mod(e, self.n_episode/10) == 0:
                self.sars.plot(np.array(self.current_observation)[:, 0:self.n_state])
            self.train_q()
            self.reset()
