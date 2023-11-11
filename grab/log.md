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
        















