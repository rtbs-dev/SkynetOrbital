__author__ = 'Thurston Sexton, Max Yi Ren'
import numpy as np
import itertools
import sklearn

class QLearning:
    def __init__(self, arg):
        self.n_episode = arg.n_episode
        self.alpha = arg.alpha # learning rate
        self.gamma = arg.gamma # depreciation of reward
        self.eps = arg.eps
        self.state_space = arg.state_space # state and action space can be multidimensional
        self.action_space = arg.action_space
        self.n_state = self.state_space.shape[0]
        self.n_action = self.action_space.shape[0]
        self.sars = arg.sars # environment model (state, action) -> state
        self.sars_set = [] # store visited state-action-reward-state data
        self.predict_q = sklearn.gaussian_process.GaussianProcess(theta0=0.1, thetaL=.001, thetaU=1.) # isotropic gaussian
        self.init_state = arg.init_state
        self.current_state = self.init_state
        self.current_action = []

        self.design = arg.design

    # pick action with highest q value based on state
    def best_action(self, s):
        feasible_action = self.get_feasible_action(s)
        Q = np.zeros((feasible_action.size,))
        for i, action in enumerate(feasible_action):
            Q[i] = self.predict_q(self.current_state, action)
        return max(Q), feasible_action[np.where(Q==max(Q))[0]]

    # get list of feasible actions from current state
    def get_feasible_action(self, state):
        # TODO: implement constraints on actions for given state
        return np.array(itertools.product(*self.action_space))

    # train q predictor from updated saq tuple
    def train_q(self):
        s = self.sars_set[:,range(0,self.n_state)]
        a = self.sars_set[:,range(self.n_state,self.n_state+self.n_action)]
        r = self.sars_set[:,self.n_state+self.n_action]
        ss = self.sars_set[:,range(self.n_state+self.n_action+1,self.n_state*2+self.n_action+1)]
        q = self.update_q(s,a,r,ss)
        self.predict_q.fit(np.concatenate((s,a),axis=1),q)

    # update q for sa couple and current q predictor
    def update_q(self, s, a, r, ss):
        # TODO: need to initialize predict_q so that it spits random stuff out at the beginning
        q = self.predict_q.predict(np.concatenate((s,a),axis=1)) # current q
        qq = np.zeros((q.size,)) # best q for ss
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
        self.current_state = self.init_state
        self.current_action = []

    # run an episode, currently do not update q model online
    def run(self):
        while self.current_state:
            self.current_action = self.e_greedy(self.current_state) # take action
            r, s = self.sars(self.current_state,self.current_action, self.design) # get reward and new state
            self.sars_set.append([self.current_state, self.current_action, r, s])
            self.current_state = s

    # main function
    def learn(self):
        for e in np.arange(self.n_episode):
            self.run()
            self.train_q()



# function run(){
# 	state = {'distance':car_pos9, 'speed':vehSpeed, 'time':Math.round(cTime*100)/100};
# //	state = {'distance':car_pos9, 'speed':vehSpeed};
# 	egreedy(step);// call step after finding the action with egreedy
# }
#
# function egreedy(step){
# 	$.post('/getQ', state, function(response){
# 		var Q, A, E, a_miss, p, aa, ee, qq, bar;
#
# 		if(current_step == 0){
# 			Q = response.Q;
# 			A = response.A;
# 			E = response.E;
# 			a_miss = $(a_set).not(A).get();
# 			A = A.concat(a_miss);
# 			$.each(a_miss, function(){Q.push(0);E.push(0);});
# 			sortWithIndeces(Q);// sort from small to large
# 			var a_copy = [], e_copy = [];
# 			$.each(Q.sortIndices, function(i,d){a_copy.push(A[d]); e_copy.push(E[d]);});
# 			A = a_copy.slice();
# 			E = e_copy.slice();
#
# 			p = [];
# 			for(var i=0;i<m;i++){p.push(eps/m*i);}
# 			p.push(1);
# 			bar = Math.random();
# 			aa = $.grep(A, function( n, i ) {
# 				  return ( p[i+1] > bar && p[i] <= bar);});
# 			ee = $.grep(E, function( n, i ) {
# 				  return ( p[i+1] > bar && p[i] <= bar);});
# 			qq = $.grep(Q, function( n, i ) {
# 				  return ( p[i+1] > bar && p[i] <= bar);});
# 			action = aa[0];
# 			e = ee[0];
# 			q = qq[0];
# 			V = Q[Q.length-1];
# 			step();
# 		}
# 		else{
# 			step();
# 		}
# 	});
# }
#
# function step(){
# 	var old_consumption = consumption;
# 	step_game(function(){
# 		new_state = {'distance':car_pos9, 'speed':vehSpeed, 'time':Math.round(cTime*100)/100};
# //		new_state = {'distance':car_pos9, 'speed':vehSpeed};
#
# 		$.post('/getQ',new_state, function(response){
# 			var Q, A, E, a_miss, p, aa, ee, qq, bar, action_p, e_p, q_p, delta, delta_p;
# 			Q = response.Q;
# 			A = response.A;
# 			E = response.E;
# 			a_miss = $(a_set).not(A).get();
# 			A = A.concat(a_miss);
# 			$.each(a_miss, function(){Q.push(0);E.push(0);});
# 			sortWithIndeces(Q);// sort from small to large
# 			var a_copy = [], e_copy = [];
# 			$.each(Q.sortIndices, function(i,d){a_copy.push(A[d]); e_copy.push(E[d]);});
# 			A = a_copy.slice();
# 			E = e_copy.slice();
#
# 			p = [];
# 			for(var i=0;i<m;i++){p.push(eps/m*i);}
# 			p.push(1);
# 			bar = Math.random();
# 			aa = $.grep(A, function( n, i ) {
# 				  return ( p[i+1] > bar && p[i] <= bar);});
# 			ee = $.grep(E, function( n, i ) {
# 				  return ( p[i+1] > bar && p[i] <= bar);});
# 			qq = $.grep(Q, function( n, i ) {
# 				  return ( p[i+1] > bar && p[i] <= bar);});
# 			action_p = aa[0];
# 			e_p = ee[0];
# 			q_p = qq[0];
# 			V = Q[Q.length-1];
#
# 			var reward = -(Math.round((consumption-old_consumption)/3600/1000/max_batt*1000)/10)
# 			 + (car_pos9 - state.distance)
# 			 + success*100*(1+Math.round(1000-consumption/3600/1000/max_batt*1000)/10) - penalty;
#
# 			delta_p = reward + gamma*Q[Q.length-1] - q;
# 			delta = reward + gamma*Q[Q.length-1] - V;
#
# 			$.post('/updateQ', {'Q': alpha*delta, 'e_coef': gamma*lambda}, function(){
# 				q = q + alpha*delta_p;
# 				e = e+1;
# 				$.post('/saveQ', {'Q':q, 'e':e, 'distance':state.distance, 'speed':state.speed, 'time':state.time, 'action':action}, function(){
# 					if(!game_finished){
# 						current_step += 1;
#
# 						// move to the new state
# 						state = new_state;
# 						action = action_p;
# 						if(battempty && action==1){action = 0;}//cannot acc when battery is empty
# 						q = q_p + alpha*delta*e_p;
#
# 						egreedy(step);
# 					}
# 					else if(episode<max_episode){
# 						if (success){
# 							score.push((Math.round(1000-(consumption/3600/1000/max_batt*1000))/10)); // for plot_status
# 						}
# 						else{
# 							score.push(0);
# 						}
# 						$.post('/saveepisode', {'episode':episode, 'score':score[episode]}, function(){
# 							plot_status();
# 							episode += 1;
# 							restartQ();
# 						});
# 					}
# 				});
# 			});
# 		});
# 	});
# }
#
# function step_game(callback){
#     var lastTime = 0;
#     var step = function () {
#         demo.step(0);
#         var state_change = cTime>=state.time+1 || game_finished;
#         if (!state_change) {
#             requestAnimationFrame(step);
#         }
#         else{callback();}
#     };
#     step(0);
# }
#
# function restartQ(){
# 	consumption = 0;
# 	battstatus = 100;
# 	game_finished = false;
# 	if(typeof demo != 'undefined'){demo.stop();}
# 	demo = new scene();
# 	wheel1moment = Jw1;
# 	wheel2moment = Jw2;
# 	wheel1.setMoment(wheel1moment);
# 	wheel2.setMoment(wheel2moment);
# 	counter = 0;
# 	cTime = 0;
# 	vehSpeed = 0;
# 	motor2eff = 0;
# 	car_posOld = 0;
# 	var pBar = document.getElementById("pbar");
# 	pBar.value = 0;
# 	drawLandscape();
# 	success = false;
# 	car_posOld = 0;
#     car_pos = Math.round(chassis.p.x*px2m); //-9.03
#     car_pos9 = car_pos-9;
#     vehSpeed = Math.round(Math.sqrt(Math.pow(chassis.vx,2)+Math.pow(chassis.vy,2))*px2m*2.23694);
#
# 	run();
# }
