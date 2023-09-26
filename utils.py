"""
    pip install utilmy




"""
import pandas as pd, numpy as np 
from utilmy import (pd_to_file, log)


def pd_generate(nhist=10000, dirout="./hist.csv", n_loc=5, n_item=10, n_user=12):
    """
        python utils.py. pd_generate  --dirout  "./hist.csv"

    """
    log("######### df_hist history  ")
    df_hist = pd.DataFrame()
    df_hist['ts']      = np.arange(0, nhist)
    df_hist['loc_id']  = np.random.randint(0, n_loc,  nhist)
    df_hist['item_id'] = np.random.randint(0, n_item, nhist)
    df_hist['user_id'] = np.random.randint(0, n_user, nhist)

    pd_to_file(df_hist, dirout )





###################################################################################################
if __name__ == "__main__":
    import fire
    fire.Fire()





