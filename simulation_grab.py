"""Docs

   ### Test
   export pyinstrument=1
   python simulation.py   test1


   ### experiments
   export pyinstrument=0
   python simulation_grab.py  run  --cfg "config.yaml"   --T 10    --dirout ztmp/exp/ --K 2


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
        + = [0.99, 0.95, 0.9, 0.85, 0.8, 0.75, . . . , 0.75]).
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



"""
import pandas as pd, numpy as np, os,json
import fire, pyinstrument
from scipy.stats import kendalltau
from collections import defaultdict
from utilmy import (log, os_makedirs, config_load, json_save, pd_to_file, pd_read_file,
date_now)


from bandits_to_rank.opponents.grab import GRAB


def binomial_sample(p: float, size: int = 1, n: int = 1):
    return np.random.binomial(n=n, p=p, size=size)


def generate_click_data(cfg: str, T: int, dirout='data_simulation.csv'):
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


def train_grab(cfg, df, K, dirout="ztmp/"):
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


            agent.update(action_list, reward_list)

        diroutk = f"{dirout}/agent_{loc_id}/"
        os_makedirs(diroutk)
        agent.save(diroutk)
        agents.append(agent)

        diroutr = f"{dirout}/regret_{loc_id}/"
        os_makedirs(diroutr)
        json_save(regret, diroutr + "/regret.json" )


    return agents


def eval_agent(agents, df):
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



##########################################################################
def run(cfg:str="config.yaml", dirout='ztmp/exp/', T=1000, nsimul=1, K=2):    

    dt = date_now(fmt="%Y%m%d_%H%M")
    dirout2 = dirout + f"/{dt}_T_{T}"
    cfgd = config_load(cfg)

    results = {}
    for i in range(nsimul):
        df      = generate_click_data(cfg= cfg, T=T, dirout= None)
        pd_to_file(df, dirout2 + f"/data/data_simulation_{i}.csv")
        agents  = train_grab(cfg, df, K, dirout=dirout2)
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




if __name__ == "__main__":
    if os.environ.get('pyinstrument', "0") == "1":
        profiler = pyinstrument.Profiler()
        profiler.start()

        fire.Fire()
        profiler.stop()
        print(profiler.output_text(unicode=True, color=True))
    else:
        fire.Fire()


