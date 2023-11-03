#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""

 GRAB RL algo

  Explain reward training in contextual bandit algorithm.



   L items : Fixed (nb_arms)

   get_Action --> List of K items out of L items


   ### Current Algo:
      reward (list of  1 or 0) --> Update the parameters of Grab

           update(self, propositions, rewards):


  ### Next Milestone: Contextual bandit (  Q-learning )
      Reward = F( context )     

      context = (time, hour day, location)id,  ...)

      A) need to create a model (ie Random Forest or Logistic Regression)
             Model2(Xinput) --> predict_Reward = Reward_estimate

      B) Will use this Reward_estimate INSIDE the GRAB Algo.

     WHy ? 

         impression, Click, 
           1 location:   Bandit return list of K items out of L items.

         Suppose  23 locations ????   
            Solution 1 : 
               23 Bandit RL algo Independant:
                  Grab1, Grab2, .....
               --> Complicated.


         Instea of having 23 Grab models.....
          1 Grab model BUT
             Reward = F( context=  [location_id, time, ....] )

             get_Action(context=  [location_id, time, ....] )

                 Differnt action output per location_id (ie different context)














"""
from random import shuffle
from math import log
from bandits_to_rank.tools.tools import swap_full, start_up, newton
import numpy as np
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib  # Import joblib



import pandas as pd 

class GRAB:
    """
    """

    def __init__(self, nb_arms, nb_positions, T, gamma, forced_initiation=False):
        """
        Parameters
        ----------
        nb_arms
        nb_positions
        T
            number of iteration
        gamma
            periodicity of force GRAB to play the leader
        forced_initiation
            constraint on the nb_arms first iteration to play
            a permutation in order to explore each item at each position at least once

        """
        self.nb_arms = nb_arms             ### Total number of item: L
        self.nb_positions = nb_positions   ### return action list size : K < L items
        self.list_transpositions = [(0, 0)]
        self.gamma = gamma
        self.forced_initiation = forced_initiation
        self.certitude = log(T)
        self.clean()

        ####reward model.
        #self.reward_model_path = 'ztmp/random_forest_model.joblib'

        # self.reward_model = joblib.load(model_save_path)
        self.reward_model      = RandomForestClassifier(n_estimators=10, random_state=0)



    def clean(self):
        """ Clean log data. /To be ran before playing a new game. """

        # clean the log
        self.precision = 0
        self.running_t = 0
        self.extended_leader = [i for i in range(self.nb_arms)]; shuffle(self.extended_leader)
        self.list_transpositions = [(0, 0)]

        self.kappa_thetas = np.zeros((self.nb_arms, self.nb_arms))
        self.times_kappa_theta = np.zeros((self.nb_arms, self.nb_positions))
        self.upper_bound_kappa_theta = np.ones((self.nb_arms, self.nb_arms))
        self.leader_count = defaultdict(self.empty)  # number of time each arm has been the leader

    @staticmethod
    def empty(): # to enable pickling
        return 0

    def choose_next_arm(self):
        (i, j) = (0, 0)
        if self.forced_initiation and (self.running_t < self.nb_arms):
            proposition = np.array([(self.running_t+i) % self.nb_arms for i in range(self.nb_positions)])
            return proposition, 0

        elif self.leader_count[tuple(self.extended_leader[:self.nb_positions])] % self.gamma > 0:
            delta_upper_bound_max = 0
            for (k, l) in self.list_transpositions:
                item_k, item_l = self.extended_leader[k], self.extended_leader[l]
                value = - self.upper_bound_kappa_theta[item_k, k] - self.upper_bound_kappa_theta[item_l, l] \
                        + self.upper_bound_kappa_theta[item_l, k] + self.upper_bound_kappa_theta[item_k, l]
                if (value > delta_upper_bound_max):
                    (i, j) = (k, l)
                    delta_upper_bound_max = value
        proposition = np.array(swap_full(self.extended_leader, (i, j), self.nb_positions))
        return proposition, 0

    def update(self, propositions, rewards):
        """ GRAB model parameters are updated HERE


        """
        
        self.running_t += 1

        # update statistics
        self.leader_count[tuple(self.extended_leader[:self.nb_positions])] += 1
        for k in range(self.nb_positions):
            item_k = propositions[k]
            kappa_theta, n = self.kappa_thetas[item_k, k], self.times_kappa_theta[item_k, k]

            ### we use reward here
            kappa_theta, n = kappa_theta + (rewards[k] - kappa_theta) / (n + 1), n + 1
            print(kappa_theta)
            start = start_up(kappa_theta, self.certitude, n)
            upper_bound = newton(kappa_theta, self.certitude, n, start)
            self.kappa_thetas[item_k, k], self.times_kappa_theta[item_k, k] = kappa_theta, n
            self.upper_bound_kappa_theta[item_k, k] = upper_bound

        # update the leader L(n) (in the neighborhood of previous leader)
        self.update_leader()
        self.update_transition()


    def update2(self, context, actions, true_rewards, mode:str):
        """ GRAB model parameters are updated HERE

            + reward learning

                # print(context, actions, true_rewards, status)
                
                #### Predict rewards2
                ### init 
                # self.model_reward = RandomForest()
                # self.Xhisto 
                ### or batch past data fit s
                # if mode='batch_fit':
                #    y = rewards  
                #    self.Xhisto = pd.concat((self.Xhisto, pd.DataFrame(context ) )
                #    self.yhisto = pd.concat((self.yhisto, y)) 
                #    self.model_reward.fit(self.Xhisto, yhisto )

                # #### List of n_arms float  
                # rewards2 = self.model_reward.predict(context)

                ### Case real time partial fit

        """
       #  model_save_path = 'random_forest_model.joblib'
        
        if mode == 'train':
            #Training reward model on the batch size 
            X = context
            y = true_rewards
            X_train, X_val, actions_train, actions_val, rewards_train, rewards_val = train_test_split(X, actions, y, test_size=0.2)
            self.reward_model.fit(np.column_stack((X_train, actions_train)), rewards_train)
            predicted_rewards = self.reward_model.predict(np.column_stack((X_val, actions_val)))[0].tolist()

            # Evaluate the model
            #accuracy = accuracy_score(rewards_val[0], predicted_rewards)
            #precision = precision_score(rewards_val[0], predicted_rewards)
            #recall = recall_score(rewards_val[0], predicted_rewards)
            f1 = f1_score(rewards_val[0], predicted_rewards)

            #print(f'Accuracy: {accuracy}')
            #print(f'Precision: {precision}')
            #print(f'Recall: {recall}')
            print(f'F1-score: {f1}')      
            from utilmy import os_makedirs
            os_makedirs(self.reward_model_path)                  
            joblib.dump(self.reward_model, self.reward_model_path)
        

        elif mode == 'predict':
            #Using trained model for prediction 
            #if os.path.exists(model_save_path):
            try:    
                input_features    = np.concatenate((context, actions.tolist()), axis = 0).reshape(1, -1)
                predicted_rewards = self.reward_model.predict(input_features)[0].tolist()

            except Exception as e:
                print(f"model failed", e)
                predicted_rewards = true_rewards
            

            # self.running_t += 1
            ############# update GRAB model :ranking list ##########################################
            self.leader_count[tuple(self.extended_leader[:self.nb_positions])] += 1
            for k in range(self.nb_positions):
                item_k = actions[k]
                kappa_theta, n = self.kappa_thetas[item_k, k], self.times_kappa_theta[item_k, k]
                ### we use reward here
                kappa_theta, n = kappa_theta + (predicted_rewards[k] - kappa_theta) / (n + 1), n + 1
                start = start_up(kappa_theta, self.certitude, n)
                upper_bound = newton(kappa_theta, self.certitude, n, start)
                self.kappa_thetas[item_k, k], self.times_kappa_theta[item_k, k] = kappa_theta, n
                self.upper_bound_kappa_theta[item_k, k] = upper_bound

            # update the leader L(n) (in the neighborhood of previous leader)
            self.update_leader()
            self.update_transition()


    def update_leader(self):
        """

        Returns
        -------

        Examples
        -------
        >>> import numpy as np
        >>> player = GRAB(nb_arms=5, nb_positions=3, T=100)
        >>> mu_hats = np.array([[0.625, 0.479, 0.268, 0., 0.],
        ...        [0.352, 0.279, 0.139, 0., 0.],
        ...        [0.585, 0.434, 0.216, 0., 0.],
        ...        [0.868, 0.655, 0.335, 0., 0.],
        ...        [0.292, 0.235, 0.108, 0., 0.]])
        >>> player.kappa_thetas = mu_hats
        >>> mu_hats[[2, 3, 1], np.arange(3)].sum()
        1.379
        >>> mu_hats[[3, 0, 2], np.arange(3)].sum()
        1.563
        >>> mu_hats[[3, 2, 0], np.arange(3)].sum()
        1.57

        >>> player.update_leader()
        >>> player.extended_leader
        array([3, 2, 0, 1, 4])

        """
        row_ind, col_ind = linear_sum_assignment(-self.kappa_thetas)
        self.extended_leader = row_ind[np.argsort(col_ind)]

    def update_transition(self):
        pi = np.argsort(-self.kappa_thetas[self.extended_leader[:self.nb_positions], np.arange(self.nb_positions)])
        self.list_transpositions = [(0, 0)]
        pi_extended = list(pi) + ([i for i in self.extended_leader if i not in pi])
        for i in range(self.nb_arms - 1):
            if i < self.nb_positions:
                self.list_transpositions.append((pi_extended[i], pi_extended[i + 1]))
            else:
                self.list_transpositions.append((pi_extended[self.nb_positions - 1], pi_extended[i + 1]))


    def save(self, dirout):
        os.makedirs(dirout, exist_ok=True)

        mdict = {
            'nb_arms': self.nb_arms,
            'nb_positions': self.nb_positions,
            'list_transpositions': self.list_transpositions,
            'gamma': self.gamma,
            'forced_initiation': self.forced_initiation,
            'certitude': self.certitude,
            'kappa_thetas': self.kappa_thetas,  
            'times_kappa_theta': self.times_kappa_theta,
            'upper_bound_kappa_theta': self.upper_bound_kappa_theta,
            'leader_count': self.leader_count,
            'running_t': self.running_t,
            'extended_leader': self.extended_leader,

            'reward_model_path': self.reward_model_path 
        }

        with open(os.path.join(dirout, 'model_grab.pkl'), 'wb') as file:
            pickle.dump(mdict, file)


    def load(self, dirin):

            with open(os.path.join(dirin, "model_grab.pkl"), 'rb') as file:
                mdict = pickle.load(file)

            self.nb_arms      = mdict['nb_arms']
            self.nb_positions = mdict['nb_positions']
            self.list_transpositions = mdict['list_transpositions']
            self.gamma = mdict['gamma']
            self.forced_initiation = mdict['forced_initiation']
            self.certitude = mdict['certitude']
            self.kappa_thetas = mdict['kappa_thetas']
            self.times_kappa_theta = mdict['times_kappa_theta']
            self.upper_bound_kappa_theta = mdict['upper_bound_kappa_theta']
            self.leader_count     = mdict['leader_count']
            self.running_t        = mdict['running_t']
            self.extended_leader  = mdict['extended_leader']

            #### REWARD model PART
            self.reward_model_path = mdict['reward_model_path']
            self.reward_model      = self.load_rewardmodel()


    def load_rewardmodel(self):
            try:
                self.reward_model = joblib.load(model_save_path)
            except: 
                self.reward_model = RandomForestClassifier(n_estimators=10, random_state=0)




if __name__ == "__main__":
    import fire 
    fire.Fire()
    #import doctest
    #doctest.testmod()

#   python simulation.py  run2  --K 3 --name simul   --T 100     --dirout ztmp/exp/  --cfg config.yaml 
