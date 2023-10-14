"""Docs

   ### Test
   export pyinstrument=1
   python simulation.py   test1


   ### experiments
   export pyinstrument=0
   python simulation_grab.py  run  --cfg "config.yaml"   --T 10    --dirout ztmp/exp/







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


def train_grab(cfg, df, dirout="ztmp/" ):
    """
    Simulate and test a GRAB-based recommendation system using a provided dataset.

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
        agent = GRAB(nb_arms, nb_positions=2, T=T, gamma=10)

        # Iterate through the DataFrame rows and simulate game actions    
        for _, row in dfi.iterrows():
            item_id = row['item_id']
            is_clk  = row['is_clk']

            # One action :  1 full list of item_id  and reward : 1 vector of [0,..., 1 , 0 ]
            action_list, _ = agent.choose_next_arm()
            reward_list = np.where(np.arange(nb_arms) == item_id, is_clk, np.zeros(nb_arms))
            agent.update(action_list, reward_list)

        diroutk = f"{dirout}/agent_{loc_id}/"
        os_makedirs(diroutk)
        agent.save(diroutk)
        agents.append(agent)

    return agents


def eval_agent(agents, df):
    """
       List of List :
          1 loc_id --->. List of item_id. : ranked by click amount.

        T = 10, 100, 500, 1000, 5000, 10000.   --> Kendall_avg
 

    """

    def get_itemid_list(dfi):
        df2 = dfi.groupby(['item_id']).agg({'is_clk': 'sum'}).reset_index()
        df2.columns = ['item_id','n_clk']
        df2 = df2.sort_values('n_clk', ascending=False)
        return df2['item_id'].values

    dfc = df[df['is_clk'] == 1]
    dfg = dfc.groupby('loc_id').apply(lambda dfi: get_itemid_list(dfi) ).reset_index()
    dfg.columns = ['loc_id', 'list_true']

    res = defaultdict(float)
    locid_all = len(agents)
    for loc_id in range(locid_all):
        agent    = agents[loc_id]
        list_true = dfg[dfg.loc_id == loc_id ]['list_true'].tolist()
        action_list, _ = agent.choose_next_arm()  ### return 1 Single List

        #### Calcuation
        res[loc_id] = sum(1 for item in action_list if item in list_true[0][:len(action_list)]) / len(action_list)


    return res



##########################################################################
def run(cfg:str="config.yaml", dirout='ztmp/exp/', T=1000, nsimul=10):    

    dt = date_now(fmt="%Y%m%d_%H%M")
    dirout2 = dirout + f"/{dt}_T_{T}"
    cfgd = config_load(cfg)

    results = {}
    for i in range(nsimul):
        df      = generate_click_data(cfg= cfg, T=T, dirout= None)
        pd_to_file(df, dirout2 + f"/data/data_simulation_{i}.csv")
        agents  = train_grab(cfg, df, dirout=dirout2)
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


