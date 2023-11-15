
###### New model to predict reward
""" 
cl;ass myMdoel
  fit
  predict
  save
  load

Linear TS:
 model_reward.fit()  model_reward.predict()

Bayesian update
    reward = Normal_distribution( params =  mean ?, variance ? )

    fit(reward_data):
        with new data (reward data) ---> calculate NEW mean, and variance....

    Predict: 
         Sample from Normal_distribution(mean, variance) ---> generate one vector of reward.

    Prediction --> Float [ 0.2, 0.5, 0.6        ] --> binary [ 0, 1, 1]      0.2 < threshold
               --->  Classification F1 score.



Benefit: Less overfit, more stable.
 """


def test():
  ts1 = LinTS()





class LinTS:

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
            #'''This method simulates the situation where the features are changed at each time t'''
            K=self.nbArms
            d=self.dimension//2
            
            #Generation of normalized features - ||x|| = 1 and x symmetric
            X = torch.randn((K,d))
            X=torch.Tensor([transform(x) for x in X])
            self.features=X
            
    def new_MAB(self):
        #'''This method actualize the MAB problem given the new features'''
        self.update_features()
        
        return self.bandit_generator(self.features,sigma=self.sigma)

    def name(self):
        return self.strat_name
        








if __name__ == "__main__":
    import fire 
    fire.Fire()
    #import doctest
    #doctest.testmod()

#   python simulation.py  run2  --K 3 --name simul   --T 100     --dirout ztmp/exp/  --cfg config.yaml 
