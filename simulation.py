import pandas as pd
import numpy as np
import os
import fire
import pyinstrument


def binomial_sample(p: float, size: int = 1, n: int = 1):
    return np.random.binomial(n=n, p=p, size=size)


def generate_click_data(cfg, T):
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
    locations, items = list(cfg['loc_probas'].keys()), list(
        cfg['item_probas'].keys())
    for ts in range(T):
        loc_id = np.random.choice(locations)
        item_id = np.random.choice(items)
        loc_prob = cfg['loc_probas'][loc_id]
        item_prob = cfg['item_probas'][item_id][loc_id]

        is_clk = binomial_sample(item_prob*loc_prob)[0]
        data.append([ts, loc_id, item_id, is_clk])

    df = pd.DataFrame(data, columns=['ts', 'loc_id', 'item_id', 'is_clk'])
    return df


if __name__ == "__main__":
    if os.environ.get('pyinstrument', "0") == "1":
        profiler = pyinstrument.Profiler()
        profiler.start()

        fire.Fire()
        profiler.stop()
        print(profiler.output_text(unicode=True, color=True))
    else:
        fire.Fire()
