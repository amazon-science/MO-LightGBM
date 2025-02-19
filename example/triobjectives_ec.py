import os
import sys
from utils import config2dict, dict2config, log2dataframe, circle_points, sample_inverse_preferences
import numpy as np
import yaml
from itertools import product

configfile = sys.argv[1]
with open(configfile, 'r') as f:
    config = yaml.safe_load(f)
dataset = config["dataset"]
objective_type = config["lightgbm_parameters"]["objective_type"]
num_objectives = 3
# Relevant paths
config_root = os.path.dirname(os.path.abspath(configfile))

objective_type_root = os.path.join(config_root
                                   , f"results_{dataset['name']}"
                                   , f"{objective_type}"
                                   )

outfolder = os.path.join(objective_type_root, "triobjective_results")

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
params["mo_preferences"] = ','.join(['1'] * num_objectives)

for key, value in config['lightgbm_parameters'].items():
    params[key] = value

main_label = dataset['main_label']
all_labels = dataset["all_labels"]
trilabels_idx = dataset['trilabels_idx']
ignore_columns = dataset['ignore_columns']
mg_combinators = config['mg_combinators']['constraint_based']

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
    results = log2dataframe(logfile, ["Training-loss"])["Training-loss"]
    results.columns = all_labels
    baseline_data[label] = results.iloc[-1]
# -------------------------------------

nub = config['num_tradeoffs']
scriptlines = {comb: [] for comb in mg_combinators}
logs = dict()

for label_idx in trilabels_idx:
    labels = [all_labels[i] for i in label_idx]
    mo1_label, mo2_label, target = labels
    params["label_column"] = f"name:{target}"
    params["mo_labels"] = f"name:{mo1_label},{mo2_label}"
    params["ignore_column"] = "name:" + ",".join(list(set(ignore_columns).intersection(set(cols)) - {target}))
    ubs1 = np.linspace(baseline_data[mo1_label][mo1_label], baseline_data[mo2_label][mo1_label], nub)
    ubs2 = np.linspace(baseline_data[mo2_label][mo2_label], baseline_data[mo1_label][mo2_label], nub)
    ubs = list(product(ubs1,ubs2))
    logs[tuple(labels)] = []

    labels_txt = '-'.join(labels)
    for ub in ubs:
        ub_data = {"ub": list(ub)}
        for combinator in mg_combinators:
            prefix = f"{combinator}_{labels_txt}"
            pref_txt = '-'.join(['%.6f']*len(ub)) % tuple(ub)
            params["mg_combination"] = combinator
            params["mo_ub_sec_obj"] = ",".join(map(str, ub))
#            configfilename = f"{combinator}_{mo_label}-{target}_" + "{:.6}.conf".format(ub)
            configfilename = f"{prefix}_{pref_txt}.conf"
            outsubfolder = os.path.join(outfolder, combinator)
            configfile = os.path.join(outsubfolder, configfilename)
            dict2config(params, configfile)
            logfile = f"{prefix}_{pref_txt}.log"
#            logfile = f"{combinator}_{mo_label}-{target}_" + "{:.6}.log".format(ub)
            ub_data[combinator] = logfile
            scriptline = lightgbm + " config=" + configfilename + " 2>&1 | tee " + logfile
            scriptlines[combinator].append(scriptline)
        logs[tuple(labels)].append(ub_data)

runscripts = {comb: os.path.join(outfolder, comb, comb + "_run_experiment.sh") for comb in mg_combinators}
for comb in mg_combinators:
    with open(runscripts[comb], 'w') as f:
        f.write('\n'.join(scriptlines[comb]))

with open(os.path.join(outfolder, "ec_run_experiment.sh"), 'w') as f:
    for comb in mg_combinators:
        f.write(f'cd {comb}; ')
        f.write(f"sh {comb}_run_experiment.sh; ")
        f.write(f'cd ..\n')

log_results = os.path.join(outfolder, "ec_log_results.yml")
with open(log_results, 'w') as f:
    yaml.dump(logs, f)
