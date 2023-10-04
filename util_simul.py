"""Fake Data Generator
Docs::
"""
import os, sys, time, datetime, inspect, json, yaml, gc, glob
from typing import List, Optional, Tuple, Union
from box import Box
from dataclasses import dataclass
from pathlib import Path


import pandas as pd, numpy as np, scipy, fire
from scipy.stats import rankdata, truncnorm

from src.utils.utilmy_log import log, log2, log3, logw, loge
from src.utils.utilmy_base import (  pd_read_file, pd_read_file2, date_now)




#####################################################################################################################
######### User Click Generator ######################################################################################
def gaussian_sample(mean, std, size=1):
    return np.random.normal(loc=mean, scale=std, size=size)


def powerlaw_sample(mean, std, size=1, a=0.4):
    # return scipy.stats.powerlaw().rvs(a=a,loc=mean,scale=std,size=size)
    return scipy.stats.powerlaw.rvs(a, loc=mean, scale=std, size=size)


def binomial_sample(p:float, size:int=1, n:int=1):
    return np.random.binomial(n=n, p=p, size=size)


def generate_user_click(
    n_users_min_avg = 10,
    n_users_min_std = 3,
    imp_avg   = 50,     imp_var   =  5,
    ctr_avg   = 0.05,   ctr_var   =  0.02,
    tstep_avg = 60.0,   tstep_std = 40.0    
):
    """  Generate user user click data
    Docs::

           dfu   = {'dt': tstep, 'user_id': [user_id] * nimp, 'click': click}
           XX users per 60mins,
           1 user -->  imp impression, ctr% clicks, evert tstep internval
    """
    imin  = 0
    dfall = pd.DataFrame()
    for ti_min in range(0, 60 * 24, 60):

        tstart      = date_now("2021-06-01", add_mins=ti_min, returnval='unix')
        n_users_min = max(1, int(gaussian_sample(n_users_min_avg, n_users_min_std, 1)[0]))

        for i in range(imin, n_users_min):
            user_id = i
            nimp = int(powerlaw_sample(imp_avg, imp_var)[0])
            ctr  = powerlaw_sample(ctr_avg, ctr_var)[0]

            click = binomial_sample(p=ctr, n=1, size=nimp)

            tstep = gaussian_sample(mean=tstep_avg, std=tstep_std, size=nimp)
            tstep = [max(5, ti) for ti in tstep]
            tstep = tstart + np.cumsum(tstep)
            tstep = [datetime.datetime.fromtimestamp(ti) for ti in tstep]

            dfu   = {'dt': tstep, 'user_id': [user_id] * nimp, 'click': click}
            dfu   = pd.DataFrame(dfu)
            dfall = pd.concat((dfall, dfu))

        imin = i

    return dfall


def generate_item_id(df, n_item=50, n_slot=5):
    """ Generate DF with     click_itemid, click_position, click_proba
    """

    def get_itemid(x):
        return np.random.randint(0, n_item) if x == 1 else 0

    df['click_itemid'] = df['click_itemid'].apply(lambda x: get_itemid(x))

    def get_item_position(x):
        return np.random.randint(0, n_slot) if x == 1 else 0

    df['click_position'] = df['click_position'].apply(lambda x: get_item_position(x))

    def get_item_proba(x):
        return np.random.random() if x == 1 else np.random.random() * 0.5

    df['click_proba'] = df['click_proba'].apply(lambda x: get_item_proba(x))

    return df


####################################################################################################################
####################################################################################################################
def fake_create_history_click(n_user=1000, n_item=100, n_event_type=2,
                              start='2022-02-01 08:00:00', end='2022-02-01 12:00:00', freq='5S'
                              ):
    """  Generate fake historical click data
    Docs::

          df_hist :  ts, user_id, item_id,  event_type_id, event( 1/0)
          df_user  :  user_id, user_f1 (int 2 values), user_f2(int 10 values), .., user_f5 (int 5 values)
          df_item  :  item_id,  item_feat1,, item_feat2 (int 10 values), .., item_feat5 (int 10 values)
          df_event :  even_type_id,  e_feat1, e_feat3, e_feat3
    """
    coluser_cat = [('user_cat1', 2), ('user_cat2', 5)]  #### Unique values
    coluser_num = [('user_num1', 1.0), ('user_num2', 1.0)]

    colitem_cat = [('item_cat1', 2), ('item_cat2', 5)]
    colitem_num = [('item_num1', 1.0), ('item_num2', 1.0)]

    colevent_cat = [('event_cat1', 2), ('event_cat2', 3)]
    colevent_num = [('event_num1', 1.0), ('event_num2', 1.0)]

    log("######### df_hist history  ")
    df_hist = pd.DataFrame()
    df_hist['date'] = pd.date_range(start=start, end=end, freq=freq)  ### 10 secnd
    df_hist['ts'] = df_hist['date'].apply(lambda x: time.mktime(x.timetuple()))  ### unix time
    nhist = len(df_hist)

    df_hist['user_id']  = np.random.randint(0, n_user, nhist)
    df_hist['item_id']  = np.random.randint(0, n_item, nhist)
    df_hist['event_id'] = np.random.randint(0, n_event_type, nhist)

    ### Model
    df_hist['model_proba'] = np.random.rand(nhist, 1)

    ### Action   ## Click or not
    n_slot = 5
    df_hist['action'] = df_hist['model_proba'].apply(lambda x: 1 if x > 0.5 else 0)
    df_hist['position'] = np.random.randint(0, n_slot, nhist)

    log("######### df_user ")
    df_user = fake_create_Xfeature(n_user, colid='user_id', colscat=coluser_cat, colsnum=coluser_num)

    log("######### df_item ")
    df_item = fake_create_Xfeature(n_item, colid='item_id', colscat=colitem_cat, colsnum=colitem_num)

    log("######### df_event ")
    df_event = fake_create_Xfeature(n_event_type, colid='event_id', colscat=colevent_cat, colsnum=colevent_num)

    return df_hist, df_user, df_item, df_event


def fake_create_Xfeature(n, colid='user_id', colscat=None, colsnum=None):
    df_user = pd.DataFrame()
    df_user[colid] = np.arange(0, n - 1)

    for (coli, nunique) in colscat:
        df_user[coli] = np.random.randint(0, nunique - 1, n - 1)

    for (coli, nunique) in colsnum:
        df_user[coli] = np.random.rand(n - 1, 1)

    return df_user


####################################################################################################################
####################################################################################################################
def generate_sample_fromdata(df, cols: list, model_sampler=None, kernel='gaussian',
                             bandwidth=0.2, return_model=False):
    """ Fit empirical distribution and generate sample data
    Docs::

        # X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        ["gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"]
        https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/
        https://www.statsmodels.org/devel/generated/statsmodels.nonparametric.kernel_density.KDEMultivariate.html
    """
    from sklearn.neighbors import KernelDensity
    if model_sampler is None:
        model_sampler = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(df[cols].values)

    samples = model_sampler.score_samples(df[cols].values)

    if return_model:
        return samples, model_sampler
    else:
        return samples


def stat_create_empirical_distribution(narray):
    """ Create Empirical distribution from data.
    Docs::
      rv_histogram.rvs(10)  ### 10 sample.
      Args:
          narray (_type_): _description_
      Returns:
          _type_: _description_
    """
    import scipy
    hist = np.histogram(narray, bins=100)
    hist_dist = scipy.stats.rv_histogram(hist)
    return hist_dist


def pd_generate_ctr_data(N_A, N_B, p_A, p_B, days=None, control_label='A',
                         test_label='B', seed=None):
    """Returns a pandas dataframe with fake CTR data
    Docs::

        Parameters:
            N_A (int): sample size for control group
            N_B (int): sample size for test group Note: final sample size may not match N_A provided because the
                       group at each row is chosen at random (50/50).
            p_A (float): conversion rate; conversion rate of control group
            p_B (float): conversion rate; conversion rate of test group
            days (int): optional; if provided, a column for 'ts' will be included
                to divide the data in chunks of time
                Note: overflow data will be included in an extra day
            control_label (str)
            test_label (str)
            seed (int)
        Returns:
            pd.DataFrame: the generated ctr dataframe
            pd.DataFrame: summary dataframe
    """
    import scipy.stats as scs
    if seed:
        np.random.seed(seed)

    # initiate empty container
    data = []

    # total amount of rows in the data
    N = N_A + N_B

    # distribute events based on proportion of group size
    group_bern = scs.bernoulli(N_A / (N_A + N_B))

    # initiate bernoulli distributions from which to randomly sample
    A_bern = scs.bernoulli(p_A)
    B_bern = scs.bernoulli(p_B)

    for idx in range(N):
        # initite empty row
        row = {}
        # for 'ts' column
        if days is not None:
            if type(days) == int:
                row['ts'] = idx // (N // days)
            else:
                raise ValueError("Provide an integer for the days parameter.")
        # assign group based on 50/50 probability
        row['group'] = group_bern.rvs()

        if row['group'] == 0:
            # assign conversion based on provided parameters
            row['converted'] = A_bern.rvs()
        else:
            row['converted'] = B_bern.rvs()
        # collect row into data container
        data.append(row)

    # convert data into pandas dataframe
    df = pd.DataFrame(data)

    # transform group labels of 0s and 1s to user-defined group labels
    df['group'] = df['group'].apply(
        lambda x: control_label if x == 0 else test_label)

    # summary dataframe
    ab_summary = df.pivot_table(values='converted', index='group', aggfunc=np.sum)
    # add additional columns to the pivot table
    ab_summary['total'] = df.pivot_table(values='converted', index='group', aggfunc=lambda x: len(x))
    ab_summary['rate'] = df.pivot_table(values='converted', index='group')

    return df, ab_summary


#####################################################################################################
if 'utils':
    def to_float(x):
        try:
            return float(x)
        except:
            return float("NaN")


    def to_int(x):
        try:
            return int(x)
        except:
            return float("NaN")


    def is_int(x):
        try:
            int(x)
            return True
        except:
            return False


    def is_float(x):
        try:
            float(x)
            return True
        except:
            return False


################################################################################################################
###########  Fake historical Sparse Click  ######################################################################
def pd_historylist_to_csr(df: pd.DataFrame, colslist: list = None, hashSize: int = 5000, dtype=np.float32,
                          max_rec_perlist: int = 5,
                          min_rec_perlist: int = 0, sep_genre=",", sep_subgenre="/"):
    """ Creates Sparse matrix of dimensions:
            Single value  max=i+1, min=i
            ncol: hashsize * (nlist1 + nlist2 + ....)    X    nrows: nUserID
            xdf:  pd.DataFrame
                genreCol: string: "4343/4343/4545, 4343/4343/4545, 4343/4343/4545, 4343/4343/4545, 4343/4343/4545"
            colist:   list of column names containing history list
            hashSize: size of hash space
            return X: scipy.sparse.coo_matrix
    """
    import mmh3
    from scipy.sparse import coo_matrix, csr_matrix, lil_matrix

    ### Ncols = nb of col
    Xcols = hashSize * len(colslist) * (max_rec_perlist - min_rec_perlist)  # top 5 genre for each reclist

    # No. rows for sparse matrix X, N_userid
    Xrows = len(df)

    # Create zeros sparse matrix
    X = lil_matrix((Xrows, Xcols), dtype=dtype)

    bucket = 0
    ntot = 0
    for coli in colslist:
        bucket0 = bucket  ### Store
        recList = df[coli].values
        for idx, genre_list in enumerate(recList):
            if isinstance(genre_list, str): genre_list = genre_list.split(sep_genre)  ### 353/34534,  5435/4345,

            ### Iterate for each genre in the reclist and reset to base bucket0
            bucket = bucket0
            for genre in genre_list[min_rec_perlist:max_rec_perlist]:
                for subgenre in genre.split(sep_subgenre):  #### 35345/5435/345345
                    ntot = ntot + 1
                    colid = mmh3.hash(subgenre.strip(), 42, signed=False) % hashSize
                    X[(idx, bucket + colid)] = 1
                bucket += hashSize

    X = csr_matrix(X)
    log('Sparse matrix shape:', X.shape)
    log('Expected no. of Ones: ', ntot)
    log('No. of Ones in the Matrix: ', X.count_nonzero())
    return X


####################################################################################################################
if __name__ == "__main__":
    fire.Fire()



