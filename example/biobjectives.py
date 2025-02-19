import os
import sys

import pandas as pd

from utils import config2dict, dict2config, log2dataframe, circle_points, sample_inverse_preferences
import numpy as np
import yaml


configfile = sys.argv[1]
with open(configfile, 'r') as f:
    config = yaml.safe_load(f)
dataset = config["dataset"]
objective_type = config["lightgbm_parameters"]["objective_type"]

# Relevant paths
config_root = os.path.dirname(os.path.abspath(configfile))
objective_type_root = os.path.join(config_root
                                   , f"results_{dataset['name']}"
                                   , f"{objective_type}"
                                   )

outfolder = os.path.join(objective_type_root, "biobjective_results")

if not os.path.isdir(outfolder):
    os.makedirs(outfolder)
lightgbm = os.path.join(config_root, config['lightgbm_path'])

# Read sample config and make changes for single objective ranking optimization
params = config2dict(config['sample_lightgbm_config'])
params["train_data_file"] = os.path.join(config_root, dataset['train_file'])
params["valid_data_file"] = os.path.join(config_root, dataset['valid_file'])
infile = open(params["train_data_file"], 'r')
cols = infile.readline().strip().split("\t")

params["group_column"] = "name:" + dataset['query_column']

for key, value in config['lightgbm_parameters'].items():
    params[key] = value

main_label = dataset['main_label']
all_labels = dataset["all_labels"]
bilabels_idx = dataset['bilabels_idx']
ignore_columns = dataset['ignore_columns']
mg_combinators = config['mg_combinators']['preference_based']

for combinator in mg_combinators:
    outsubfolder = os.path.join(outfolder, combinator)
    if not os.path.isdir(outsubfolder):
        os.makedirs(outsubfolder)

# -------- Get baseline data -------------
baseline_folder = os.path.join(objective_type_root, f"baseline_results")
with open(os.path.join(baseline_folder, "log_results.yml"), 'r') as f:
    baseline_log_results = yaml.safe_load(f)
baseline_data = dict()
for i, label in enumerate(all_labels):
    logfile = os.path.join(baseline_folder, baseline_log_results[label])
    print(f'open {logfile}')
    results = log2dataframe(logfile, ["Training-loss"])["Training-loss"]
    results.columns = all_labels
    baseline_data[label] = results.iloc[-1]
# -------------------------------------

npref = config['num_tradeoffs']
scriptlines = {comb: [] for comb in mg_combinators}
logs = dict()
for bilabel_idx in bilabels_idx:
    bilabel = [all_labels[i] for i in bilabel_idx]
    mo_label, target = bilabel
    params["label_column"] = "name:" + target
    params["mo_labels"] = "name:" + mo_label
#    params["ignore_column"] = "name:" + ",".join(list(set(ignore_columns) - {target}))
    params["ignore_column"] = "name:" + ",".join(list(set(ignore_columns).intersection(set(cols)) - {target}))
    logs[(mo_label, target)] = []

#    min_angle = np.arctan(baseline_data[mo_label][mo_label] / baseline_data[mo_label][target])
#    max_angle = np.arctan(baseline_data[target][mo_label] / baseline_data[target][target])
    _d = np.array([baseline_data[target][mo_label], baseline_data[mo_label][target]])
    _d = _d / np.linalg.norm(_d)
    _d = np.maximum(_d, np.array([1e-6, 1e-6]))
#    print('_d = ', _d)
#    print(' tag tag = ', baseline_data[target])
    min_angle = np.arctan( (baseline_data[target][target] /_d[1]) / (baseline_data[target][mo_label] / _d[0] ))
    max_angle = np.arctan( (baseline_data[mo_label][target] /_d[1]) / (baseline_data[mo_label][mo_label] / _d[0]))
    inv_preferences = circle_points(npref, min_angle=min_angle, max_angle=max_angle)
    #preferences = 1.0 /inv_preferences
    #print("preference (before normalization) = ", inv_preferences)
#    print("before normalization")
#    for i in range(len(inv_preferences)):
#        print(inv_preferences[i] / inv_preferences[i][0])
    for i in range(len(inv_preferences)):
        inv_preferences[i] = inv_preferences[i] * _d
#    print("after normalization")
#    for i in range(len(inv_preferences)):
#        print(inv_preferences[i] / inv_preferences[i][0])

    preferences = 1.0 /inv_preferences
    for i in range(len(preferences)):
        preferences[i] = preferences[i] / np.linalg.norm(preferences[i],ord=1)

    for pref in preferences:
        pref_data = {"preferences": pref.tolist()}
        params["mo_preferences"] = ",".join([str(p) for p in pref])
        for combinator in mg_combinators:
            #
            if combinator == "pmtl":
                preffilename = f"{combinator}_{mo_label}-{target}_" + "{:.6}-{:.6}.tsv".format(pref[0], pref[1])
                preffilepath = os.path.join(outfolder, combinator, preffilename)
                pd.DataFrame(preferences).to_csv(preffilepath, header=None, index=None, sep="\t")
                params["mo_pmtl_preferencefile_path"] = preffilepath
        #
            params["mg_combination"] = combinator
            outsubfolder = os.path.join(outfolder, combinator)
            configfilename = f"{combinator}_{mo_label}-{target}_" + "{:.6}-{:.6}.conf".format(pref[0], pref[1])
            configfile = os.path.join(outsubfolder, configfilename)
            dict2config(params, configfile)
            logfile = f"{combinator}_{mo_label}-{target}_" + "{:.6}-{:.6}.log".format(pref[0], pref[1])
            pref_data[combinator] = logfile
            scriptline = lightgbm + " config=" + configfilename + " 2>&1 | tee " + logfile
            scriptlines[combinator].append(scriptline)
        logs[(mo_label, target)].append(pref_data)




runscripts = {comb: os.path.join(outfolder, comb, comb + "_run_experiment.sh") for comb in mg_combinators}

for comb in mg_combinators:
    with open(runscripts[comb], 'w') as f:
        f.write('\n'.join(scriptlines[comb]))

with open(os.path.join(outfolder, "run_experiment.sh"), 'w') as f:
    for comb in mg_combinators:
        f.write(f'cd {comb}; ')
        f.write(f"sh {comb}_run_experiment.sh; ")
        f.write(f'cd ..\n')

log_results = os.path.join(outfolder, "log_results.yml")

with open(log_results, 'w') as f:
    yaml.dump(logs, f)
