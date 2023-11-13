python simulation.py  run2  --K 3 --name simul   --T 100     --dirout ztmp/exp/  --cfg config.yaml 

Result while training
##### Start Simul  
F1-score: 0.0
F1-score: 0.7499999999999999
F1-score: 0.8571428571428571
F1-score: 0.8235294117647058
F1-score: 0.8571428571428571
F1-score: 0.888888888888889
F1-score: 0.9032258064516129
F1-score: 0.9189189189189189
F1-score: 0.9268292682926829
F1-score: 0.9333333333333333
F1-score: 0.92
F1-score: 0.9056603773584906
F1-score: 0.9152542372881356
F1-score: 0.9032258064516129
F1-score: 0.9090909090909091
F1-score: 0.9166666666666666
F1-score: 0.9210526315789475
F1-score: 0.9268292682926829
F1-score: 0.9302325581395349
F1-score: 0.9333333333333333
F1-score: 0.9375
F1-score: 0.9400000000000001
F1-score: 0.9433962264150945
F1-score: 0.9357798165137614
F1-score: 0.9380530973451328


###############################################################
Please copy paste the ouput of your runs 



Task List:

1) check the stat.csv and copy paste here
   make you understand the probaility output of _stats.csv
    ### Stats
    dfg = df.groupby(['loc_id', 'item_id']).agg({'is_clk': 'sum', 'ts': 'count'}).reset_index()
    dfg.columns = ['loc_id', 'item_id', 'n_clk', 'n_imp']
    dfg['ctr'] = dfg['n_clk'] / dfg['n_imp']


         item_probas:
            # locid:   [itemid0, itemid1, itemid3, ...]
            #0:  [ 0.1, 0.2 , 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95 ]
            # idx     0     1      2    3     4    5      6
            0:  [ 0.01, 0.03 , 0.04, 0.05, 0.8, 0.9, 0.95 ]  #### proba of click  for location 0 
            1:  [ 0.95, 0.9 , 0.8, 0.05, 0.04, 0.03, 0.01 ]  #### proba of click  for location 1 

        ##### In location 0 : 
        # For location 0 :    Top-3 best item_id are. [ 6, 5,4 ] with click proba  [ 0.95, 0.9, 0.8 ]
        # For location 1 :    Top-3 best item_id are. [ 0, 1, 3 ] with click proba [ 0.95, 0.9, 0.8 ]




2) Test code this model: in separate file
  with fake tensor data.
    new file.

Xfake data.  --- To check the code is running.

context_ts/experiments/banditBaselines.py
    
class LinTS:
    """Linear Thompson Sampling Strategy"""
    def __init__(self,X,nu,bandit_generator=None,reg=1,sigma=0.5,name='LinTS'):
        self.features=X
        self.reg=reg
        (self.nbArms,self.dimension) = X.size()
        self.nu=nu
        self.sigma=sigma
        self.strat_name=name
        self.bandit_generator=bandit_generator
        self.clear()

    def clear(self):
        with torch.no_grad():
            # initialize the design matrix, its inverse, 
            # the vector containing the sum of r_s*x_s and the least squares estimate
            self.t=1
            self.Design = self.reg*torch.eye(self.dimension)
            self.DesignInv = (1/self.reg)*torch.eye(self.dimension)
            self.Vector = torch.zeros((self.dimension,1))
            self.thetaLS = torch.zeros((self.dimension,1)) # regularized least-squares estimate
        
    
    def chooseArmToPlay(self):
        with torch.no_grad():
            N=torch.distributions.multivariate_normal.MultivariateNormal(self.thetaLS.view(-1),(self.nu*self.nu)*self.DesignInv)
            theta_tilda=N.sample()
            return torch.argmax(torch.matmul(self.features,theta_tilda)).item()

    def receiveReward(self,arm,reward):
        with torch.no_grad():
            x = self.features[arm,:].view((self.dimension,1)) #column vector
            self.Design = self.Design + torch.matmul(x,x.T) 
            self.Vector = self.Vector + reward*x
            # online update of the inverse of the design matrix
            omega=torch.matmul(self.DesignInv,x)
            self.DesignInv= self.DesignInv-torch.matmul(omega,omega.T)/(1+torch.matmul(x.T,omega).item())
            # update of the least squares estimate 
            self.thetaLS = torch.matmul(self.DesignInv,self.Vector)
            self.t+=1
        
    def update_features(self):
        with torch.no_grad():
            '''This method simulates the situation where the features are changed at each time t'''
            K=self.nbArms
            d=self.dimension//2
            
            #Generation of normalized features - ||x|| = 1 and x symmetric
            X = torch.randn((K,d))
            X=torch.Tensor([transform(x) for x in X])
            self.features=X
            
    def new_MAB(self):
        '''This method actualize the MAB problem given the new features'''
        self.update_features()
        
        return self.bandit_generator(self.features,sigma=self.sigma)

    def name(self):
        return self.strat_name
        



ts 	itemid_list   	itemid_clk    	context 	action_list 	rwd_best 	rwd_actual 	rwd_list 	predicted_reward 	regret 	regret_bad_cum 	regret_ratio        	rwd_actual_cum 	regret_cum 	regret_bad_cum_cum 	regret_ratio_cum
 0 	0,1,2,3,4,5,6 	0,0,0,0,1,0,1 	      0 	2,0,5       	       2 	       0.0 	0,0,0    	0,0,1            	   2.0 	           2.0 	0.2857142857142857  	           0.0 	       2.0 	               2.0 	0.2857142857142857
 1 	0,1,2,3,4,5,6 	0,0,0,0,1,1,1 	      0 	0,1,5       	       3 	       1.0 	0,0,1    	0,0,1            	   2.0 	           4.0 	0.2857142857142857  	           1.0 	       4.0 	               6.0 	0.5714285714285714
 2 	0,1,2,3,4,5,6 	0,0,0,0,1,1,1 	      0 	5,1,0       	       3 	       1.0 	1,0,0    	1,0,0            	   2.0 	           6.0 	0.2857142857142857  	           2.0 	       6.0 	              12.0 	0.8571428571428571
 3 	0,1,2,3,4,5,6 	0,0,0,1,1,1,1 	      0 	0,2,5       	       4 	       1.0 	0,0,1    	0,0,1            	   3.0 	           9.0 	0.32142857142857145 	           3.0 	       9.0 	              21.0 	1.1785714285714286
 4 	0,1,2,3,4,5,6 	0,0,0,0,1,0,1 	      0 	1,0,5       	       2 	       0.0 	0,0,0    	0,0,1            	   2.0 	          11.0 	0.3142857142857143  	           3.0 	      11.0 	              32.0 	1.4928571428571429
 5 	0,1,2,3,4,5,6 	0,0,0,0,0,1,1 	      0 	0,3,5       	       2 	       1.0 	0,0,1    	0,0,1            	   1.0 	          12.0 	0.2857142857142857  	           4.0 	      12.0 	              44.0 	1.7785714285714285
 6 	0,1,2,3,4,5,6 	0,0,0,0,1,1,1 	      0 	1,0,5       	       3 	       1.0 	0,0,1    	0,0,1            	   2.0 	          14.0 	0.2857142857142857  	           5.0 	      14.0 	              58.0 	2.064285714285714
 7 	0,1,2,3,4,5,6 	0,0,0,0,1,1,1 	      0 	5,1,0       	       3 	       1.0 	1,0,0    	1,0,0            	   2.0 	          16.0 	0.2857142857142857  	           6.0 	      16.0 	              74.0 	2.3499999999999996
 8 	0,1,2,3,4,5,6 	0,0,0,0,1,1,1 	      0 	0,4,5       	       3 	       2.0 	0,1,1    	0,0,1            	   1.0 	          17.0 	0.2698412698412698  	           8.0 	      17.0 	              91.0 	2.6198412698412694
 9 	0,1,2,3,4,5,6 	0,0,0,0,1,1,1 	      0 	0,6,5       	       3 	       2.0 	0,1,1    	0,1,1            	   1.0 	          18.0 	0.2571428571428571  	          10.0 	      18.0 	             109.0 	2.8769841269841265
 0 	0,1,2,3,4,5,6 	1,1,1,0,0,0,0 	      1 	2,0,1       	       3 	       3.0 	1,1,1    	0,0,0            	   0.0 	          18.0 	0.23376623376623376 	          13.0 	      18.0 	             127.0 	3.11075036075036
 1 	0,1,2,3,4,5,6 	1,0,1,1,0,0,0 	      1 	0,1,2       	       3 	       2.0 	1,0,1    	1,0,1            	   1.0 	          19.0 	0.2261904761904762  	          15.0 	      19.0 	             146.0 	3.3369408369408364
 2 	0,1,2,3,4,5,6 	1,1,1,0,0,0,0 	      1 	0,3,2       	       3 	       2.0 	1,0,1    	1,1,1            	   1.0 	          20.0 	0.21978021978021978 	          17.0 	      20.0 	             166.0 	3.5567210567210563
 3 	0,1,2,3,4,5,6 	1,1,1,0,0,0,0 	      1 	0,3,2       	       3 	       2.0 	1,0,1    	1,1,1            	   1.0 	          21.0 	0.21428571428571427 	          19.0 	      21.0 	             187.0 	3.7710067710067707
 4 	0,1,2,3,4,5,6 	1,1,1,0,0,0,0 	      1 	0,3,2       	       3 	       2.0 	1,0,1    	1,1,1            	   1.0 	          22.0 	0.20952380952380953 	          21.0 	      22.0 	             209.0 	3.9805305805305804
 5 	0,1,2,3,4,5,6 	1,1,0,0,0,0,0 	      1 	0,3,2       	       2 	       1.0 	1,0,0    	1,0,1            	   1.0 	          23.0 	0.20535714285714285 	          22.0 	      23.0 	             232.0 	4.185887723387723
 6 	0,1,2,3,4,5,6 	1,1,0,0,0,0,0 	      1 	0,2,3       	       2 	       1.0 	1,0,0    	1,1,0            	   1.0 	          24.0 	0.20168067226890757 	          23.0 	      24.0 	             256.0 	4.387568395656631
 7 	0,1,2,3,4,5,6 	1,1,1,0,0,0,0 	      1 	0,4,2       	       3 	       2.0 	1,0,1    	1,0,1            	   1.0 	          25.0 	0.1984126984126984  	          25.0 	      25.0 	             281.0 	4.5859810940693295
 8 	0,1,2,3,4,5,6 	1,1,1,0,0,0,0 	      1 	0,5,2       	       3 	       2.0 	1,0,1    	1,1,1            	   1.0 	          26.0 	0.19548872180451127 	          27.0 	      26.0 	             307.0 	4.781469815873841
 9 	0,1,2,3,4,5,6 	1,1,0,0,0,0,0 	      1 	0,5,2       	       2 	       1.0 	1,0,0    	1,0,1            	   1.0 	          27.0 	0.19285714285714287 	          28.0 	      27.0 	             334.0 	4.974326958730984


action_list	ts
0,3,2	4
0,5,2	2
1,0,5	2
5,1,0	2
0,1,2	1
0,1,5	1
0,2,3	1
0,2,5	1
0,3,5	1
0,4,2	1
0,4,5	1
0,6,5	1
2,0,1	1
2,0,5	1











