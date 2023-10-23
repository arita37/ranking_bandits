import pandas as pd
import subprocess
import os


def find_convergence_index(lst):
    for i in range(len(lst)):
        if len(set(lst[i:])) == 1:
            return i
    return i


def get_latest_folder_in_directory(directory_path):
    folders = [os.path.join(directory_path, folder) for folder in os.listdir(
        directory_path) if os.path.isdir(os.path.join(directory_path, folder))]
    if not folders:
        return None

    latest_folder = max(folders, key=os.path.getctime)
    return latest_folder


results = []
for _ in range(20):
    bash_command = "python simulation_grab.py  run2  --K 3 --name simul   --T 20000     --dirout ztmp/exp/  --cfg config.yaml"
    subprocess.run(bash_command, shell=True, stdout=subprocess.PIPE,
                   stderr=subprocess.PIPE, text=True)
    folder = "ztmp\\exp"
    results_path = get_latest_folder_in_directory(folder)
    results_path = os.path.join(
        results_path, "sim0\\0\\metrics\\simul_metrics.csv")
    action_lists = pd.read_csv(results_path, sep="\t")["action_list"]
    action_lists = action_lists.apply(
        lambda x: [int(action) for action in x.split(',')])
    action_lists = action_lists.apply(lambda x: sorted(x))
    action_lists = action_lists.apply(lambda x: tuple(x)).values
    res = find_convergence_index(action_lists)
    results.append(res)

print(results)
