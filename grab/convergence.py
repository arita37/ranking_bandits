import pandas as pd
import subprocess
import os
import math

PROBA = 1. / math.comb(7, 3)


def is_convergent(df, factor):
    return df.values.max()/df.values.sum()>factor*PROBA

    

def get_latest_folder_in_directory(directory_path):
    folders = [os.path.join(directory_path, folder) for folder in os.listdir(
        directory_path) if os.path.isdir(os.path.join(directory_path, folder))]
    if not folders:
        return None

    latest_folder = max(folders, key=os.path.getctime)
    return latest_folder



def main():
    T = 1000
    bash_command = "python simulation_grab.py  run2  --K 3 --name simul   --T 20000     --dirout ztmp/exp/  --cfg config.yaml"
    subprocess.run(bash_command, shell=True, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True)
    folder = "ztmp\\exp"
    results_path = get_latest_folder_in_directory(folder)
    results_path = os.path.join(
        results_path, "sim0\\0\\metrics\\simul_metrics.csv")
    df_experiment = pd.read_csv(results_path, sep="\t")

    for ts in range(40, T):

        df = df_experiment[df_experiment['ts']<=ts]
        df["action_list"] = df["action_list"].apply(
            lambda x: tuple(sorted([int(action) for action in x.split(',')])))
        df = df.groupby('action_list').count()['ts']
        if is_convergent(df, factor=1):
            print(f'convergence at index = {ts}')
            print(df)
            return ts


if __name__=="__main__":
    main()
