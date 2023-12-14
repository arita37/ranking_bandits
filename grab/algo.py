"""





"""

import numpy as np
from random import shuffle
#from math import log
import math 
from bandits_to_rank.tools.tools import swap_full, start_up, newton
import numpy as np
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
import os
import pickle
# from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib  # Import joblib

# from utilmy import log, os_makedirs  #confusion between math log and utilmy log 
from utilmy import (os_makedirs, log)  
import pandas as pd 

def test2():
    n_arms = 5
    nb_positions =5
    T = 10 
    gamma = 0.1
    bandit = newBandit(n_arms=n_arms, nb_positions=nb_positions, gamma=gamma, T=T)
    contexts =[  np.random.rand(1, nb_positions) for i in range(0, n_arms) ]
    print(contexts)
    top_k_list, reward_all_items = bandit.choose_next_arm(contexts)
    log('Top k items', top_k_list)
    log('Updating Batch Reward list and context')
    bandit.reward_model.update_batch(reward_all_items, contexts)
    log('-------------Complete-----------')
    bandit.save_rewardmodel()
    log('Reward model save')
    

class newBandit:
    """
    """

    def __init__(self, n_arms, nb_positions, T, gamma, forced_initiation=False,
                 reward_model_path="ztmp/reward_model/reward_model11.joblib"):
        """
        Parameters
        ----------
        n_arms
        nb_positions
        T
            number of iteration
        gamma
            periodicity of force GRAB to play the leader
        forced_initiation
            constraint on the n_arms first iteration to play
            a permutation in order to explore each item at each position at least once

        """
        self.n_arms = n_arms             ### Total number of item: L
        self.nb_positions = nb_positions   ### return action list size : K < L items

        self.R = 3 ### exploration list
        self.gamma = 1.0                     


        self.forced_initiation = forced_initiation


        ####reward model Load
        self.reward_model_path = reward_model_path
        
        # self.reward_model = RandomForestClassifier(n_estimators=10, random_state=0)
        self.load_rewardmodel()
        self.clean()


    def clean(self):
        """ Clean log data. /To be ran before playing a new game. """

        # clean the log
        self.precision = 0
        self.running_t = 0
        self.extended_leader = [i for i in range(self.n_arms)]; shuffle(self.extended_leader)
        self.list_transpositions = [(0, 0)]


    @staticmethod
    def empty(): # to enable pickling
        return 0


    def choose_next_arm(self, Xcontext:pd.DataFrame):
        """ 
        """
        ## Predict the average reward = [ 0.4, 0.2,  0.5  ]
        reward_all_items = self.reward_model.predict_rewards_float(Xcontext) ### predict Average reward for each item

        
        ###Algo to Select Best List:  list of item_id   len(topk_list) =  self.nb_positions 
        ## before it was GRAB, 
        topk_list        = self.topk_predict_list(reward_all_items)   
        return topk_list, reward_all_items


    def topk_predict_list(self, reward_list_float):
        """ 
        Be careful

                A=   [ (item_id, reward_value) , .... ]

        As initialized to   [0, M-R]      Best reward value items

                            [M-R+1,  M]    items : exploration.

        """
        gamma = self.gamma

        A  = np.arange(0, self.n_arms) ### all items
        # log('All item', reward_list_float)
        # concatenated_array = np.concatenate(reward_list_float.reshape(-1, 1), axis=0)
        concatenated_array = np.array(reward_list_float)
        # log('Input all reward list', concatenated_array)
        # Get the indices that would sort the concatenated array in descending order
        indices_descending = np.argsort(concatenated_array)
        # log('Input all reward list', reward_list_float)
        log('After sorting item id on the basis of best reward', indices_descending)
        # Use the indices to sort the original list of arrays
        As = [reward_list_float[i] for i in indices_descending] ### As reward

        # As_exploit list on the basis df best/highest reward value 
        As_exploit = As[: self.nb_positions - self.R]
        As_explore = As[self.nb_positions - self.R:]
        As_expoit_item_id  = indices_descending[: self.nb_positions - self.R]
        As_explore_item_id = indices_descending[self.nb_positions - self.R:]
        
        # log(f'\nExploitation reward: {As_exploit} and item id {As_expoit_item_id}')
        # log(f'\nExpolaration : {len(As_explore)}, so the expolaration rewrds {As_explore}')
        # log('Items need to explore',As_explore_item_id, "\n")

        # reward_dict = {f'{i}': array for i, array in enumerate(reward_list_float)}
        new_array_list =  []
        #### Add R remaining items by exploration 
        for i in As_explore_item_id:
            # Aneg represents the remaining items for exploration
            Aneg = np.setdiff1d(As, As_exploit)
            # Initialize Plist with zeros
            Plist = np.zeros(len(Aneg))
            for u in range(0, len(Aneg) ):
                # Find the argmax(a belongs to Aneg) r(Xs, a)
                imax = np.argmax(Aneg)
                # Calculate Plist based on exploration 
                if u != imax:
                    Plist[u] = np.abs(1 / len(Aneg) + gamma * (As[imax] - As[u]))
                else:
                    Plist[u] = np.abs(1 - np.sum(Aneg))
                    
            # Sample from Plist to select an item for exploration
            # Normalize Plist to ensure probabilities sum to 1

            Plist_normalized = abs(Plist) / np.sum(Plist)
            sample = ([[np.random.choice(Aneg, size=1, p=Plist_normalized, replace=True)]])
            
            # Update As_exploit by adding the sampled item
            new_array_list.append(sample)
        # Flatten the array and get the indices that would sort it in descending order
        indices_descending = np.argsort(np.array(new_array_list).flatten())[::-1]
        #mapping the orginal reward array index to sorted reward array index which is obtained from exploration
        # map = {0:As_explore_item_id[0], 1:As_explore_item_id[1], 2:As_explore_item_id[2], 3:As_explore_item_id[3], 4:As_explore_item_id[4], 5:As_explore_item_id[5], 6:As_explore_item_id[6]}
        # indices = [map.get(i) for i in indices_descending]
        size_of_As_explore_item_id = len(As_explore_item_id)

        # Create a mapping based on the size of As_explore_item_id
        map = {i: As_explore_item_id[i] for i in range(size_of_As_explore_item_id)}

        # Ensure that the index is within the valid range
        indices = [map.get(i, None) for i in indices_descending]
        items = np.append(As_expoit_item_id, np.array(indices))
        # Get the top k items
        top_items = items[:self.nb_positions]
        #sanity check for Before exploration items have to same for top items values. 
        assert np.array_equal(As_expoit_item_id, top_items[ :self.nb_positions-self.R])
        # log('all items after TS model ', items)
        return items #### [ 7,5 , 8, 1, ]


    def update(self, mode:str, dftrain:pd.DataFrame=None):
        """ Update reward model and topk predict list model
            dftrain
            dfi['y']          = rwd_list       ### list size is L-items (ie all the items)
            dfi['context-x1'] = Xcontext_list  ### list of Array(1, dvector)
            dfi['actions']    = action_list    ### list of itemid 

        Issue :

           action_list : List of arm_id ( = itemid) ,  len(action_list) = K

           BUt there are L itemid in total and  L >> K
             and we update ALL the L items. in update_batch

             Need to reindex the list



        """

        ##### Update reward model only  #################################
        dftrain.to_csv('dftrain.csv')
        reward_list, context_list = [], [] 
        for i in range(0, self.n_arms):
        #### reward model update
            reward_list0 = dftrain['y'][i]
            context_list0 = dftrain['context-x1'][i]
            reward_list.append(reward_list0)
            context_list.append(context_list0)   
        self.reward_model.update_batch(i, reward_list, context_list)



        #### topk rank list update: ####################################
        ## this model does not need any update, because it computes online at prediction.
        ## State of the model is Fixed :   


    def save(self, dirout):
        os.makedirs(dirout, exist_ok=True)

        mdict = {
            'n_arms': self.n_arms,
            'nb_positions': self.nb_positions,

            'forced_initiation': self.forced_initiation,

            'running_t': self.running_t,

            'reward_model_path': self.reward_model_path, 
            'reward_model': self.reward_model
        }

        with open(os.path.join(dirout, 'model_grab.pkl'), 'wb') as file:
            pickle.dump(mdict, file)

    def load(self, dirin):

            with open(os.path.join(dirin, "model_grab.pkl"), 'rb') as file:
                mdict = pickle.load(file)

            self.n_arms      = mdict['n_arms']
            self.nb_positions = mdict['nb_positions']
            
            self.forced_initiation = mdict['forced_initiation']

            self.leader_count     = mdict['leader_count']
            self.running_t        = mdict['running_t']

            #### REWARD model PART
            self.reward_model_path = mdict['reward_model_path']
            self.reward_model      = self.load_rewardmodel()

    def load_rewardmodel(self):
        try:
            self.reward_model = joblib.load(self.reward_model_path)
        except: 
            log("cannot load, using default ")
            self.reward_model =  LinearTS(self.n_arms, self.nb_positions)

    def save_rewardmodel(self):
        os_makedirs(self.reward_model_path)
        joblib.dump(self.reward_model, self.reward_model_path)



################################################################################################
def test1():
        # Define the number of arms and the dimension of the context
        n_arms = 5
        d = 3

        model  = LinearTS(n_arms, d)

        contexts =[  np.random.rand(1, d) for i in range(0, n_arms) ]
        reward_list = model.predict_rewards_float(contexts)
        ###########
        rewards  = np.random.rand(n_arms) ### Real reward
        contexts =[  np.random.rand(1, d) for i in range(0, n_arms) ]

        model.update_batch(rewards, contexts)





class LinearTS:
    """ 
       Predict mean reward for each item.
          using Bayesian sampling :
             m(k) = Normal( mu_k, var_k )

         d: size of embedding :
            2 locations.
                --> 10 arms



    """
    def __init__(self, n_arms, d, alpha=2.0):
        self.n_arms = n_arms
        self.d = d ## context dim
        self.alpha = alpha

        self.B      = [ np.eye(2) for i in range(0, self.n_arms) ]
        self.mu_hat = [ np.zeros((d, 2)) for i in range(0, self.n_arms) ]
        self.f      = [ np.zeros((d, 2)) for i in range(0, self.n_arms) ]


    def update_batch(self, index:int, reward_list:list, context_list:np.array):
        """
         context_list : list of array(1, dvector) (1 array pe arm)
         reward_list : list of 0 or 1 (per each arm)

        """
        log('Before Batch Update ')
        # log(f'B : {self.B}, Mu Hat: {self.mu_hat} , F : {self.f}')
        for i_arm, (reward, context) in enumerate( zip(reward_list, context_list)):
            if len(context_list)==0:
                pass  
            
            context = np.array(context[0]).reshape(-1, 1)
            reward = np.array(reward).reshape(-1, 1)
            self.B[i_arm] += context @ context.T
            self.f[i_arm] += (reward * context.T)
            self.mu_hat[i_arm]    =  self.f[i_arm] @ np.linalg.inv(self.B[i_arm])

        log('\nAfter Batch Update: ', i_arm)
        # log(f'B : {self.B}, Mu Hat: {self.mu_hat} , F : {self.f}')


    def get_arm(self, contexts):
        sample_reward = self.predict_rewards_float( contexts)    
        return np.argmax(sample_reward)


    def predict_rewards_float(self, contexts:list):
        """
         Location is fixed at each step

          contexts =[      ### size 10 : 10 arms
               [1,0],  ## one encoding of location. [1,0]. OR [0,1]
               [1,0],
               ...

               [1,0]
          ]


          

            mu_hat[i] = np.zeros((d, 1))     d=2 
            (self.mu_hat[i] @ context)[0,0]
            ipython or jupyter in vscode 

            Rewardn == Normal_Distribution(mean, variance)
              --> sample from the distribution.

              

        """
        sample_rewards = []
        for i in range(self.n_arms):          
            ### contexts : List of numpy array(1, M)    , len(list) = n_arms
            context = contexts[i].reshape(1, -1)    ### dimension (1,d)
            mean = (self.mu_hat[i] @ context.T)[0,0]
            var  = self.alpha * np.sqrt(context @ self.B[i] @ context.T )[0,0]
            sample_reward = np.random.normal(mean, var)
            sample_rewards.append(sample_reward)
        return sample_rewards

if __name__ == "__main__":
    import fire 
    fire.Fire()
    #import doctest
    #doctest.testmod()

#   python simulation.py  run2  --K 3 --name simul   --T 100     --dirout ztmp/exp/  --cfg config.yaml 
