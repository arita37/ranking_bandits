"""Docs

   ### Test
   export pyinstrument=1
   python simulation.py   test1


   ### experiments
   export pyinstrument=0
   python simulation.py  run  --cfg "config.yaml"   --T 1    --dirout ztmp/exp/







"""
import pandas as pd
import numpy as np
import os,json
import fire
import pyinstrument
from bandits_to_rank.opponents.top_rank import TOP_RANK
from utilmy import (log, os_makedirs, config_load, json_save)
from scipy.stats import kendalltau
from collections import defaultdict

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

    for loc_id for locations:
        item_probas = list(cfg['item_probas'][loc_id].values())
        for ts in range(T):

            ## Is click 1/0 
            is_clk = binomial_sample(item_prob)[0]

            ### which item has been clicked/shown
            item_id     = np.random.choice(items, p=item_probas)

            # item_prob   = cfg['item_probas'][loc_id][item_id]
            data.append([ts, int(loc_id), int(item_id), is_clk])

    df = pd.DataFrame(data, columns=['ts', 'loc_id', 'item_id', 'is_clk'])
    df.to_csv(dirout, index=False)
    return df


def train_toprank(cfg, df, dirout="ztmp/" ):
    """
    Simulate and test a TOP_RANK-based recommendation system using a provided dataset.

    Args:
    - cfg (str): Path to the configuration file containing dataset information and other settings.

    Returns:
    None
    """
    
    cfg = config_load(cfg)
    # df = pd.read_csv(cfg['dataframe_csv'])

    nb_arms = len(df['item_id'].unique())
    loc_id_all = len(df['loc_id'].unique())
    discount_factors = [1,1,1]
    T = len(df)

    players=[]
    #### for each location we simulate the bandit optimizer (ie list of items)
    for loc_id in range(loc_id_all):
        dfi = df[df['loc_id'] == loc_id ]
        player = TOP_RANK(nb_arms, T=T, discount_factor=discount_factors)

        # Iterate through the DataFrame rows and simulate game actions    
        for _, row in dfi.iterrows():
            item_id = row['item_id']
            is_clk  = row['is_clk']

            # One action :  1 full list of item_id  and reward : 1 vector of [0,..., 1 , 0 ]
            action_list, _ = player.choose_next_arm()
            reward_list = np.where(np.arange(nb_arms) == item_id, is_clk, np.zeros(nb_arms))
            player.update(action_list, reward_list)

        diroutk = f"{dirout}/player_{loc_id}/"
        os_makedirs(diroutk)
        player.save(diroutk)
        players.append(player)

    return players


def evaluate_ranking_kendall(players, df, nsample=10):
    """
       List of List :
          1 loc_id --->. List of item_id. : ranked by click amount.

        T = 10, 100, 500, 1000, 5000, 10000.   --> Kendall_avg

              kendall goes to 1.0 if algo is correct.  

    """

    def get_itemid_list(dfi):
        df2 = dfi.groupby(['item_id']).agg({'is_clk': 'sum'}).reset_index()
        df2.columns = ['item_id','n_clk']
        df2 = df2.sort_values('n_clk', ascending=False)
        return df2['item_id'].values

    dfc = df[df['is_clk'] == 1]
    dfg = dfc.groupby('loc_id').apply(lambda dfi: get_itemid_list(dfi) ).reset_index()
    dfg.columns = ['loc_id', 'list_true']

    ## sampling of the kendall
    res = {}
    locid_all = len(players)
    for loc_id in range(locid_all):
        player    = players[loc_id]
        list_true = dfg[dfg.loc_id == loc_id ]['list_true'].tolist()
        res[loc_id] = []
        for _ in range(nsample):
           
           action_list, _ = player.choose_next_arm()  ### return 1 Single List
           kendall_tau, _ = kendalltau(list_true, action_list)
           res[loc_id].append(kendall_tau)

    return res

##########################################################################
def test1():
    ### pytest
    # Generate a sample cfg dictionary and T value
    cfg = {
    "loc_probas": {
        "0": 0.5,
        "1": 0.3,
        "2": 0.2
    },

    "item_probas":{
        "0":{
            "0": 0.5,
            "1": 0.2,
            "2": 0.3
        },
        "1":{
            "0": 0.1,
            "1": 0.6,
            "2": 0.3
        },
        "2": {
            "0": 0.2,
            "1": 0.1,
            "2": 0.7
        }
    }
    
    }
    T = 1000

    # Call the generate_click_data function
    df = generate_click_data(cfg, T)
    assert len(df)>0



##########################################################################
def run(cfg:str="config.yaml", dirout='ztmp/exp/', T=1000):    

    dircsv = 'data_simulation.csv'
    scores = defaultdict(list)
    df      = generate_click_data(cfg= cfg, T=T, dirout= dircsv)
    players = train_toprank(cfg, df)
    kdict   = evaluate_ranking_kendall(players, df, 1000)
    dirout2 = os.path.join(dirout, 'T_'+str(T))
    json_save(kdict, dirout2 + "/result.json" )

    #for key, values in kdict.items():
    #    mean = sum(values) / len(values)
    #    scores[key].append(mean)
    #    print(f"mean for key {key} = {mean})

    #json_save(scores, dirout + "/result.json" )




if __name__ == "__main__":
    if os.environ.get('pyinstrument', "0") == "1":
        profiler = pyinstrument.Profiler()
        profiler.start()

        fire.Fire()
        profiler.stop()
        print(profiler.output_text(unicode=True, color=True))
    else:
        fire.Fire()


