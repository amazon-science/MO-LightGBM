import os
import sys
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

#outfolder = os.path.join(config_root, f"biobjective_results_{dataset['name']}")
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
params["mo_preferences"] = '1,1'

for key, value in config['lightgbm_parameters'].items():
    params[key] = value

main_label = dataset['main_label']
all_labels = dataset["all_labels"]
bilabels_idx = dataset['bilabels_idx']
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

for bilabel_idx in bilabels_idx:
    bilabel = [all_labels[i] for i in bilabel_idx]
    mo_label, target = bilabel
    params["label_column"] = "name:" + target
    params["mo_labels"] = "name:" + mo_label
    #params["ignore_column"] = "name:" + ",".join(list(set(ignore_columns) - {target}))
    params["ignore_column"] = "name:" + ",".join(list(set(ignore_columns).intersection(set(cols)) - {target}))
#    ubs = np.linspace(baseline_data[mo_label][mo_label], baseline_data[target][mo_label], nub)
    ubs = np.linspace(baseline_data[target][mo_label], baseline_data[mo_label][mo_label], nub)
    logs[(mo_label, target)] = []
    for ub in ubs.tolist():
        ub_data = {"ub": ub}
        for combinator in mg_combinators:
            params["mg_combination"] = combinator
            params["mo_ub_sec_obj"] = ",".join([str(ub)])
            configfilename = f"{combinator}_{mo_label}-{target}_" + "{:.6}.conf".format(ub)
            outsubfolder = os.path.join(outfolder, combinator)
            configfile = os.path.join(outsubfolder, configfilename)
            dict2config(params, configfile)

            logfile = f"{combinator}_{mo_label}-{target}_" + "{:.6}.log".format(ub)
            ub_data[combinator] = logfile
            scriptline = lightgbm + " config=" + configfilename + " 2>&1 | tee " + logfile
            scriptlines[combinator].append(scriptline)
        logs[(mo_label, target)].append(ub_data)

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
