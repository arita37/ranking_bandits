"""Docs

### install
    pip install -r ppip/pip.txt


### experiments

   ### Version 2 : this the one we focus on
   export pyinstrument=0
   python simulation.py  run  --cfg "config.yaml"   --T 20    --dirout ztmp/exp/ --K 3




   #### Old not working. 
      ####python simulation.py  run  --cfg "config.yaml"   --T 10    --dirout ztmp/exp/ --K 2


### Description:

   RL bandit algorithm called GRAB.

    This file run a ENV simulation.
      1) We generate into a csv file the (actions, rewards)
             generate_click_data2

             Website : Amazon : many items to display (K items)


               actions:  At each time step, we display on a website a list of K-items  (Amazon..) 
                         1 action == 1 List of K items out of L items.

               Rewards:  Users can click or not on 1 or many items

            Feed rewards --> Agent/Player and generate new action.

            ts,                loc_id,            item_id,                           is_clk
            timestamp :int    webpage_id          item_id Displayed(CocaCola)        is click: 1/0

            Assume loc_id =0 ONLY (only one webpage)
            0,0,1,0
            1,0,0,1
            2,0,0,0


      2) We use the csv file as ENVIRONNMENT values
            Reward are PRE-GENERATED ( is_clk == reward) following some probability distribution
            == Easier to validate.


         and RUN the bandit simulation :
                train_grab(cfg, df, K, dirout="ztmp/")

                Standard RL:

                for ti in range(0,T):
                  action_itemlist    = get_action()  ### list of item_id
                  reward_clklist     = get_reward(action_itemlist)
                  metrics = calcmetrics(reward,....) 

              dfmetrics.to_csv( .... )







   
    Simulation is running OK, no issues

    ### Today
    1) I need you understand the problem context 
         and the simulation.py  ---> Intuition is correct.


    ##### AFTER
    2) I need you improve the simulation code with my requirements.
          Yes, some reward change, 
               some optimization, ...
            
    3) Some part of the models ( I will explain)         
            



   ### Test
   export pyinstrument=1
   python simulation.py   test1


     Dataset simulation:

        K =  [0.99, 0.95, 0.9, 0.85, 0.8, 0.75, . . . , 0.75]
        K = 1 all the time.

        theta = P(click on item_i / Location_K)
        Ptotal = theta * Kpositin

        Each step t:
            Possible multiple click SINCE
              since item_id some Proba_i. (position, item i)
                Indepdnant bernoulli with Pi.

            Total rewardValue = [0  ,   TopK ]  (up to the size of the return list)    
              because all item can be clicked (possible)


        1 Location : Search Page 1 Search result
              K items displayed
              --> calculate the number of click within than K displayed items.

        

        Our side :
               Each step t:
                  Total Reward : O or 1 

    
       Our simulation :
              Bernoulli: isClick. Proba

              Softmax between items : Sum(proba_item) = 1.0. --> Do not allow to have mutiple.
                  Distribute among item_id with proba_i






        papers
        We use two types of datasets: a simulated one for which we
        set the values for κ and θ and a real one, where parameters
        are inferred from real life logs of Yandex search engine
        (Yandex, 2013). Let’s remind that θi
        is the probability for
        the user to click on item i when it observes this item, and
        κk is the probability for the user to observe position k.
        Simulated data allow us to test GRAB in extreme situations. We consider L = 10 items, K = 5 positions, and κ = [1, 0.75, 0.6, 0.3, 0.1]. The range of values for θ is either close to zero (θ
        − = [10−3
        , 5.10−4
        ,
        10−4
        , 5.10−5
        , 10−5
        , 10−6
        , . . . , 10−6
        ]), or close to one
        (θ

        Real data contain the logs of actions toward the Yandex
        search engine: 65 million search queries and 167 million
        hits (clicks). Common use of this database in the bandit
        setting consists first in extracting from these logs the parameters of the chosen real model, and then in simulating
        users’ interactions given these parameters (Lattimore et al.,
        2018). We use Pyclick library (Chuklin et al., 2015) to infer
        the PBM parameters of each query with the expectation
        maximization algorithm. This leads to θi values ranging
        from 0.070 to 0.936, depending on the query. Similarly to
        (Lattimore et al., 2018), we look at the results averaged on
        the 10 most frequent queries, while displaying K = 5 items
        among the L = 10 most attractive ones



#### Expriments


   python simulation.py  run  --K 3 --name simul   --T 100     --dirout ztmp/exp/  --cfg config.yaml 


   python simulation.py  run  --K 3 --name simul   --T 10000    --dirout ztmp/exp/  --cfg config.yaml 


   python simulation.py  run  --K 3 --name simul5   --T 50000    --dirout ztmp/exp/  --cfg config.yaml 




"""
import pandas as pd, numpy as np, os,json, subprocess
import fire, pyinstrument
from scipy.stats import kendalltau
from collections import defaultdict
from box import Box
import math
from utilmy import (log, os_makedirs, config_load, json_save, pd_to_file, pd_read_file,
date_now, load_function_uri, glob_glob)


from bandits_to_rank.opponents.grab import GRAB

def log3(*s):
    if os.environ.get('debug', "") == "1":
        print(*s, flush=True)

#########################################################################################
######### Version 2 : Multiple Clicks ###############################################################
def generate_click_data2(cfg: str, name='simul', T: int=None, dirout='data_simulation.csv'):
    """
    Generate a dataframe with sampled items and locations with binomial sampling

    Args:
    - cfg: A dictionary containing location_id and their probabilities, item_id and their conditional probabilities
    - T: number of timestamps

    Returns:
    - dataframe: A pandas DataFrame with the following columns:
        - ts: Timestamp (int)
        - loc_id: Location ID (int)
        - item_id: Item ID (int)
        - is_clk: Binary reward (1 for click, 0 for no click) sampled from the given probabilities. (int)

       Mutiple rows :  item_id, is_clk

    """
    cfg0 = config_load(cfg) if isinstance(cfg, str) else cfg
    cfg1 = cfg0[name]

    T    = cfg1['T']  if T is None else T
    cfgd = cfg1['env_probas'] 
    locations = list(cfgd['loc_probas'].keys())
    log(cfgd)

    ### Generate Simulated Click 
    data = []
    for loc_id in locations:
        item_probas = list(cfgd['item_probas'][loc_id])
        for ts in range(T):

            #### Check if each itemis was clicked or not : indepdantn bernoulli.
            for item_id, pi in  enumerate(item_probas): 
                is_clk    = binomial_sample(pi)[0]
                data.append([ts, int(loc_id), int(item_id), is_clk])
    df = pd.DataFrame(data, columns=['ts', 'loc_id', 'item_id', 'is_clk'])

    ### Stats
    dfg = df.groupby(['loc_id', 'item_id']).agg({'is_clk': 'sum', 'ts': 'count'}).reset_index()
    dfg.columns = ['loc_id', 'item_id', 'n_clk', 'n_imp']
    dfg['ctr'] = dfg['n_clk'] / dfg['n_imp']


    if dirout is not None:
        pd_to_file(df, dirout, index=False, show=1)
        pd_to_file(dfg, dirout.replace(".csv", "_stats.csv"), index=False, show=1)        
    return df


def train_grab(cfg,name='simul', df:pd.DataFrame=None, K=10, dirout="ztmp/"):
    """
    Simulate and test a GRAB-based recommendation system using a provided dataset. 
    Compute the regret at each iteration

    Args:
    - cfg (str): config

         python simulation.py  run  --cfg config.yaml --name 'simul'    --T 10    --dirout ztmp/exp/ --K 2

         python simulation.py  run  --cfg config.yaml --name 'simul'    --T 10000    --dirout ztmp/exp/ --K 2


       itemid_list : Displayed item at time step ts
       itemid_clk :  1 (click) or 0 for items in Displayed items
 

    """ 
    BATCH_SIZE = 4   
    cfg0 = config_load(cfg) if isinstance(cfg, str) else cfg
    cfg1 = cfg0[name]

    cc = Box({})

    ### ENV Setup
    cc.n_item_all = len(df['item_id'].unique())
    cc.loc_id_all = len(df['loc_id'].unique())
    cc.nrows      = len(df)

    ### Agent Setup
    agent_uri   = cfg1['agent'].get('agent_uri', "bandits_to_rank.opponents.grab:GRAB" )
    agent_pars  = cfg1['agent'].get('agent_pars', {} )

    agents=[]
    action_lst, reward_lst, context = [], [], []
    #### for each location: a New bandit optimizer
    for loc_id in range(cc.loc_id_all):

        ### Env: Flatten simulation data per time step
        dfi         = df[df['loc_id'] == loc_id ]
        dfg         = dfi.groupby(['ts']).apply( lambda dfi :  dfi['item_id'].values  ).reset_index()
        dfg.columns = ['ts', 'itemid_list' ]
        dfg['itemid_clk'] = dfi.groupby(['ts']).apply( lambda dfi :   dfi['is_clk'].values  )    ##. 0,0,01
        log('\n#### Simul data \n', dfg[[ 'ts', 'itemid_list', 'itemid_clk'   ]])


        log("\n#### Init New Agent ")
        agent_pars['T']       = len(dfg)      ## Correct T 
        agent_pars['nb_arms'] = cc.n_item_all ## Correct                
        agentClass = load_function_uri(agent_uri)
        agent      = agentClass(**agent_pars)
        cc.agent_pars = agent_pars        
        # agent = GRAB(**agent_pars)
        log(agent)    

        ### Metrics
        dd = {}
        log("\n##### Start Simul  ")
        regret_sum = 0
        regret_bad_cum = 0
        for t in range(0, len(dfg)):

            # Return One action :  1 full list of item_id  to be Displayed
            action_list, _ = agent.choose_next_arm()

            #### ENV reward / clk 
            itemid_imp = dfg['itemid_list'].values[t]
            itemid_clk = dfg['itemid_clk' ].values[t]

            rwd_best             = np.sum( itemid_clk )   ### All Clicks               
            rwd_actual, rwd_list = rwd_sum_intersection( action_list, itemid_imp, itemid_clk,)
            regret               = rwd_best - rwd_actual   #### Max Value  K items
            regret_bad_cum += len(itemid_imp)  # Update regret_bad_cum
            regret_sum += regret    #update regret sum 
            regret_ratio = regret_sum / regret_bad_cum  #regret ratio calculation
            context.append(loc_id)
            action_lst.append(action_list)
            reward_lst.append(rwd_list)
            # agent.update(action_list, rwd_list)
            agent.update2([loc_id], action_list, rwd_list, False)
            if len(action_lst) % BATCH_SIZE == 0:
                agent.update2(context, action_lst, reward_lst, True)
                # print(action_lst, reward_lst)
            dd = metrics_add(dd, 'action_list',   action_list)
            dd = metrics_add(dd, 'rwd_best',      rwd_best    )
            dd = metrics_add(dd, 'rwd_actual',    rwd_actual    )
            dd = metrics_add(dd, 'rwd_list',      rwd_list    )
            dd = metrics_add(dd, 'regret',        regret    )
            dd = metrics_add(dd, 'regret_bad_cum',  t * len(itemid_imp)   )   #### Worst case  == Linear
            dd = metrics_add(dd, 'regret_ratio',        regret_ratio    )   #add regret ratio
        ### Exp Params 
        log(cc)
        json_save(cc, f"{dirout}/{loc_id}/params.json")    

        log("###### Metrics Save ###########") 
        df = metrics_create(dfg, dd)
        
        log(df[[ 'rwd_best' ,  'rwd_actual', 'regret_cum', 'regret_bad_cum', 'regret_ratio']])
        diroutr = f"{dirout}/{loc_id}/metrics"
        pd_to_file(df, diroutr + "/simul_metrics.csv", index=False, show=1, sep="\t" )

        dfs = df.groupby('action_list').agg({'ts': 'count'}).reset_index().sort_values('ts', ascending=0) 
        log('action Final\n', df[[ 'action_list' ]])
        pd_to_file(dfs, diroutr + "/simul_metrics_stats.csv", index=False, show=1, sep="\t" )

        log("###### Agent Save ###########") 
        diroutk = f"{dirout}/{loc_id}/agent"
        os_makedirs(diroutk)
        agent.save(diroutk)
        agents.append(agent)
        log(diroutk)

    return agents



def rwd_sum_intersection( action_list,  itemid_list,  itemid_clk, n_item_all=10 ):
    """ 
       action_list :  List of Top-K itemid actually dispplayed

       itemid_list:   List of all item_id
       itemid_clk:    List  of 0 / 1 (is clicked), same size than itemid_list

       Return Sum(reward if itemid is inside action_list ) Only for action_list


    """
    reward_sum  = 0.0    ### Sum( click if itemid in action_list ) for this time step.
    reward_list = []
    for itemk in action_list:
         idx        = list( itemid_list).index( itemk)   ## Find index
         rwdi       = itemid_clk[idx]
         reward_sum = reward_sum + rwdi       ## Check if this itemid was clicked.  
         reward_list.append(rwdi)

    return reward_sum, reward_list



def run(cfg:str="config.yaml", name='simul', dirout='ztmp/exp/', T=1000, nsimul=1, K=2):    
    """ Run Simulation 

    """
    dt = date_now(fmt="%Y%m%d_%H%M")
    dirout2 = dirout + f"/{dt}_T{T}"
    # cfg0    = config_load(cfg)


    for i in range(nsimul):
        dirouti = f"{dirout2}/sim{i}"
        df      = generate_click_data2(cfg= cfg, name=name, T=T, 
                                       dirout= dirouti + f"/data/df_simul_{i}.csv")
        train_grab(cfg, name, df, K=K, dirout=dirouti)






##########################################################################################
#######  Run with Contextual added to Reward learning ####################################
def run2(cfg:str="config.yaml", name='simul', dirout='ztmp/exp/', T=1000, nsimul=1, K=2):    
    """ Run Simulation 

         python simulation.py  run2  --cfg config.yaml --name 'simul'    --T 10    --dirout ztmp/exp/ --K 2

         python simulation.py  run2  --cfg config.yaml --name 'simul'    --T 10000    --dirout ztmp/exp/ --K 2


    """
    dt = date_now(fmt="%Y%m%d_%H%M")
    dirout2 = dirout + f"/{dt}_T{T}"
    # cfg0    = config_load(cfg)


    for i in range(nsimul):
        dirouti = f"{dirout2}/sim{i}"
        df      = generate_click_data2(cfg= cfg, name=name, T=T, 
                                       dirout= dirouti + f"/data/df_simul_{i}.csv")
        train_grab2(cfg, name, df, K=K, dirout=dirouti)


def train_grab2(cfg,name='simul', df:pd.DataFrame=None, K=10, dirout="ztmp/"):
    """
    Simulate and test a GRAB-based recommendation system using a provided dataset. 
    Compute the regret at each iteration

    Args:
    - cfg (str): config


       itemid_list : Displayed item at time step ts
       itemid_clk :  1 (click) or 0 for items in Displayed items
 

    """ 
    BATCH_SIZE = 4   
    cfg0 = config_load(cfg) if isinstance(cfg, str) else cfg
    cfg1 = cfg0[name]

    cc = Box({})

    ### ENV Setup
    cc.n_item_all = len(df['item_id'].unique())
    cc.loc_id_all = len(df['loc_id'].unique())
    cc.nrows      = len(df)

    ### Agent Setup
    agent_uri   = cfg1['agent'].get('agent_uri', "bandits_to_rank.opponents.grab:GRAB" )
    agent_pars  = cfg1['agent'].get('agent_pars', {} )

    agents=[]
    action_lst, reward_lst, context = [], [], []
    #### for each location: a New bandit optimizer
    for loc_id in range(cc.loc_id_all):

        ### Env: Flatten simulation data per time step
        dfi         = df[df['loc_id'] == loc_id ]
        dfg         = dfi.groupby(['ts']).apply( lambda dfi :  dfi['item_id'].values  ).reset_index()
        dfg.columns = ['ts', 'itemid_list' ]
        dfg['itemid_clk'] = dfi.groupby(['ts']).apply( lambda dfi :   dfi['is_clk'].values  )    ##. 0,0,01
        log('\n#### Simul data \n', dfg[[ 'ts', 'itemid_list', 'itemid_clk'   ]])


        log("\n#### Init New Agent ")
        agent_pars['T']       = len(dfg)      ## Correct T 
        agent_pars['nb_arms'] = cc.n_item_all ## Correct                
        agentClass = load_function_uri(agent_uri)
        agent      = agentClass(**agent_pars)
        cc.agent_pars = agent_pars        
        # agent = GRAB(**agent_pars)
        log(agent)    

        #### Training Data  #####################################
        nstep_train = 2000 ## maximum size of training data
        n_item      = cc.n_item_all


        ### Metrics
        dd = {}
        log("\n##### Start Simul  ")
        regret_sum = 0
        regret_bad_cum = 0
        dftrain = pd.DataFrame() #### contains all histo
        for t in range(0, len(dfg)):

            # Return One action :  1 full list of item_id  to be Displayed
            action_list, _ = agent.choose_next_arm()

            #### ENV reward / clk 
            itemid_imp = dfg['itemid_list'].values[t]
            itemid_clk = dfg['itemid_clk' ].values[t]

            rwd_best             = np.sum( itemid_clk )   ### All Clicks               
            rwd_actual, rwd_list = rwd_sum_intersection( action_list, itemid_imp, itemid_clk,)
            regret               = rwd_best - rwd_actual   #### Max Value  K items
            regret_bad_cum      += len(itemid_imp)  # Update regret_bad_cum
            regret_sum          += regret    #update regret sum 
            regret_ratio         = regret_sum / regret_bad_cum  #regret ratio calculation


            context.append(loc_id)
            action_lst.append(action_list)
            reward_lst.append(rwd_list)


            ### Build historical train because RForest has no partial fit...
            dfi = pd.dataframe()
            dfi['y']          = rwd_list ### list size is L-items (ie all the items)
            dfi['context-x1'] = loc_id   ### only 1 features


            dftrain = pd.concat(( dfi, dftrain))             
            ### be careful of the cut off size should be mutiple of n_item.
            dftrain = dftrain.iloc[:nstep_train * n_item,:] ### only keep ntime_step

          
            #### Using Reward Learning from dynamic Context ###################
            if len(action_lst) % BATCH_SIZE == 0:
                # agent.update2(context, action_lst, reward_lst, mode='train')
                agent.update2(dftrain, mode='train_reward') ### Update Reward Model + Grab Model
                # print(action_lst, reward_lst)

            #agent.update2([loc_id], action_list, rwd_list, mode='predict')
            agent.update2( dfi, mode='use_reward_model')  ## Only update Grab model


            ####### Metrics ###################################################    
            dd = metrics_add(dd, 'action_list',   action_list)
            dd = metrics_add(dd, 'rwd_best',      rwd_best    )
            dd = metrics_add(dd, 'rwd_actual',    rwd_actual    )
            dd = metrics_add(dd, 'rwd_list',      rwd_list    )
            dd = metrics_add(dd, 'regret',        regret    )
            dd = metrics_add(dd, 'regret_bad_cum',  t * len(itemid_imp)   )   #### Worst case  == Linear
            dd = metrics_add(dd, 'regret_ratio',        regret_ratio    )   #add regret ratio
        ### Exp Params 
        log(cc)
        json_save(cc, f"{dirout}/{loc_id}/params.json")    

        log("###### Metrics Save ###########") 
        df = metrics_create(dfg, dd)
        
        log(df[[ 'rwd_best' ,  'rwd_actual', 'regret_cum', 'regret_bad_cum', 'regret_ratio']])
        diroutr = f"{dirout}/{loc_id}/metrics"
        pd_to_file(df, diroutr + "/simul_metrics.csv", index=False, show=1, sep="\t" )

        dfs = df.groupby('action_list').agg({'ts': 'count'}).reset_index().sort_values('ts', ascending=0) 
        log('action Final\n', df[[ 'action_list' ]])
        pd_to_file(dfs, diroutr + "/simul_metrics_stats.csv", index=False, show=1, sep="\t" )

        log("###### Agent Save ###########") 
        diroutk = f"{dirout}/{loc_id}/agent"
        os_makedirs(diroutk)
        agent.save(diroutk)
        agents.append(agent)
        log(diroutk)

    return agents









###############################################################################
def run_convergence(dirin= "ztmp/exp", T=100, K=3, name='simul', nsimul=2, factor=1):
    """ Check Tmin before convergence

        python simulation.py run_convergence  --dirin ztmp/exp/ --nsimul 20


    """
    def is_convergent(df, factor):
        return df.values.max()/df.values.sum()>factor

    results = []
    for _ in range(nsimul):
        #bash_command = "python simulation.py  run  --K 3 --name simul   --T 20000     --dirout ztmp/exp/  --cfg config.yaml"
        #subprocess.run(bash_command, shell=True, stdout=subprocess.PIPE,
        #            stderr=subprocess.PIPE, text=True)
        PROBA = 1. / math.comb(7, 3)

        run(K=K, name=name, T=T, dirout= dirin)
        flist  = list(sorted(glob_glob( dirin + "/**/simul_metrics.csv")))
        df_exp = pd_read_file(flist[-1], sep="\t")
        for ts in range(25, T):
            df = df_exp.loc[df_exp['ts']<=ts].copy()
            df["action_list"] = df["action_list"].apply(
                lambda x: tuple(sorted([int(action) for action in x.split(',')])))
            df = df.groupby('action_list').count()['ts']
            
            if is_convergent(df, factor=factor*PROBA):
                results.append(ts)
                break
        log(results)


################################################################################
def to_str(vv, sep=","):
    if isinstance(vv, str): return vv 
    return sep.join( [ str(x) for x in vv])

def metrics_add(dd, name, val):
    if name not in dd:
        dd[name] =[  val ] 
    else:
        dd[name].append(  val )
    return dd


def metrics_create(dfg, dd:dict ):
    df = pd.DataFrame(dd)
    df = pd.concat((dfg, df), axis=1) ### concat the simul

    for ci in df.columns:
        x0 = df[ci].values[0]
        if isinstance(x0,list) or isinstance(x0, np.ndarray ):
           df[ci] = df[ci].apply(lambda x: to_str(x))

        if isinstance(x0, float):
           df[ ci + '_cum'] = df[ci].cumsum()
    return df




##########################################################################################
####### utiles
def binomial_sample(p: float, size: int = 1, n: int = 1):
    return np.random.binomial(n=n, p=p, size=size)



if __name__ == "__main__":
    if os.environ.get('pyinstrument', "0") == "1":
        profiler = pyinstrument.Profiler()
        profiler.start()

        fire.Fire()
        profiler.stop()
        print(profiler.output_text(unicode=True, color=True))
    else:
        fire.Fire()










######################################################################################################
################ Version 1 : Not working Only one click per time step ################################
def zz_generate_click_data(cfg: str, T: int, dirout='data_simulation.csv'):
    """
    Generate a dataframe with sampled items and locations with binomial sampling

    Args:
    - cfg: A dictionary containing location_id and their probabilities, item_id and their conditional probabilities
    - T: number of timestamps

    Returns:
    - dataframe: A pandas DataFrame with the following columns:
        - ts: Timestamp (int)
        - loc_id: Location ID (int)
        - item_id: Item ID (int)
        - is_clk: Binary reward (1 for click, 0 for no click) sampled from the given probabilities. (int)
    """
    data = []
    cfg0 = config_load(cfg)
    cfg  = json.loads(cfg0['simul']['probas']) ### load the string

    locations = list(cfg['loc_probas'].keys())
    items     = list(cfg['item_probas'][locations[0]].keys())
    #loc_probas = list(cfg['loc_probas'].values())

    for loc_id in locations:
        item_probas = list(cfg['item_probas'][loc_id].values())
        for ts in range(T):

            ### which item has been clicked/impression
            item_id   = np.random.choice(items, p=item_probas)

            ## Is click 1/0 
            item_prob = cfg['item_probas'][loc_id][item_id]
            is_clk    = binomial_sample(item_prob)[0]

            data.append([ts, int(loc_id), int(item_id), is_clk])

    df = pd.DataFrame(data, columns=['ts', 'loc_id', 'item_id', 'is_clk'])
    if dirout is not None:
        df.to_csv(dirout, index=False)
    return df



def zz_train_grab(cfg, df, K, dirout="ztmp/"):
    """
    Simulate and test a GRAB-based recommendation system using a provided dataset. 
    Compute the regret at each iteration

    Args:
    - cfg (str): Path to the configuration file containing dataset information and other settings.

    Returns:
    None
    """
    
    cfg = config_load(cfg)

    nb_arms    = len(df['item_id'].unique())
    loc_id_all = len(df['loc_id'].unique())
    T          = len(df)

    agents=[]
    action_lst, reward_lst = [], []
    #### for each location we simulate the bandit optimizer (ie list of items)
    for loc_id in range(loc_id_all):
        dfi   = df[df['loc_id'] == loc_id ]
        agent = GRAB(nb_arms, nb_positions=K, T=T, gamma=10)
        regret = {}
        cumulative_expected_reward = 0
        cumulative_reward = 0
        reward_total_t = []
        # Iterate through the DataFrame rows and simulate game actions    
        for t, row in dfi.iterrows():
            item_id = row['item_id']
            is_clk  = row['is_clk']

            # One action :  1 full list of item_id  and reward : 1 vector of [0,..., 1 , 0 ]
            action_list, _ = agent.choose_next_arm()

            ##### Total reward = 1 if item_id in action_list[:topk]
            rt = 1 if item_id in action_list and is_clk >0 else 0
            reward_total_t.append(rt)

            #### Granular reward
            reward_list    = np.where(np.arange(nb_arms) == item_id, is_clk, np.zeros(nb_arms))

            if is_clk:
                cumulative_expected_reward += 1
                cumulative_reward += sum(reward_list[action_list[i]] for i in range(len(action_list)))
                regret[t] = cumulative_expected_reward-cumulative_reward

            action_lst.append(action_list)
            reward_lst.append(reward_list)
            
            if len(action_lst) % 2 == 0:
                agent.update2(action_lst, reward_lst)
        # agent.update( action_lst, reward_lst)
        print(action_lst, reward_lst)
        diroutk = f"{dirout}/agent_{loc_id}/"
        os_makedirs(diroutk)
        agent.save(diroutk)
        agents.append(agent)

        diroutr = f"{dirout}/regret_{loc_id}/"
        os_makedirs(diroutr)
        json_save(regret, diroutr + "/regret.json" )


    return agents


def zz_eval_agent(agents, df):
    """
    Evaluate Bandit Algorithm Agents for Item Ranking
    
    Args:
    - agents (list): List of bandit algorithm agents.
    - df (DataFrame): User interaction data with 'loc_id', 'item_id', and 'is_clk'. Must contain at least one is_clk = 1 per loc_id


    Returns:
    - res (dict): Evaluation results with CTR for each agent (per location).

    Description:
    Evaluates bandit algorithm agents' performance in ranking items. Calculates Click-Through Rate (CTR) for recommendations. 

    """

    def get_itemid_list(dfi):
        df2 = dfi.groupby(['item_id']).agg({'is_clk': 'sum'}).reset_index()
        df2.columns = ['item_id','n_clk']
        df2 = df2.sort_values('n_clk', ascending=False)
        return df2['item_id'].values

    dfc = df[df['is_clk'] == 1]
    dfg = dfc.groupby('loc_id').apply(lambda dfi: get_itemid_list(dfi) ).reset_index()
    dfg.columns = ['loc_id', 'list_true']

    res = {}
    locid_all = len(agents)
    for loc_id in range(locid_all):
        agent    = agents[loc_id]
        list_true = dfg[dfg.loc_id == loc_id ]['list_true'].tolist()
        action_list, _ = agent.choose_next_arm()  ### return 1 Single List

        #### Calcuation
        if loc_id not in res: 
            res[loc_id] = {}

        mm= sum(1 for item in action_list if item in list_true[0][:len(action_list)]) / len(action_list)
        res[loc_id] = mm

    return res




def zz_run(cfg:str="config.yaml", dirout='ztmp/exp/', T=1000, nsimul=1, K=5):    

    dt = date_now(fmt="%Y%m%d_%H%M")
    dirout2 = dirout + f"/{dt}_T_{T}"
    cfgd = config_load(cfg)

    results = {}
    for i in range(nsimul):
        df      = generate_click_data(cfg= cfg, T=T, dirout= None)
        pd_to_file(df, dirout2 + f"/data/data_simulation_{i}.csv")
        agents  = train_grab(cfg, df, K=K, dirout=dirout2)
        kdict   = eval_agent(agents, df)

        for k,v in kdict.items():
            if k not in results: results[k] = []
            results[k].append( v ) 
    
    res2 = {}
    for k,vlist in results.items():
        res2[k] = np.mean(vlist)
    
    metrics = {'ctr_avg_per_item' : res2,
               'config': cfgd}
    json_save(metrics, dirout2 + "/metric.json" )

