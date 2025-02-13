import os
from utils import config2dict, dict2config, circle_points, log2dataframe
import numpy as np
import pickle as pkl
import json

mg_combinators = ["e_constraint",
                  # "ec_mgda", 
                 ]

# Relevant paths
cwd = os.getcwd()
outfolder = os.path.join(cwd, "biobjective_results")
runscripts = {comb: os.path.join(outfolder, comb + "_run_experiment.sh") for comb in mg_combinators}
lightgbm = os.path.join(cwd, os.pardir, "LightGBM", "lightgbm")
log_results = os.path.join(outfolder, "ec_log_results.pkl")
sampleconfig = os.path.join(cwd, "sample_config.conf")
trainfile = os.path.join(cwd, "datasets", "inverted_qs_train.tsv")
validfile = os.path.join(cwd, "datasets", "inverted_qs_valid.tsv")

# Read sample config and make changes for single objective ranking optimization
params = config2dict(sampleconfig)
params["train_data_file"] = trainfile
params["valid_data_file"] = validfile
params["mo_preferences"] = "1,1"
# params["num_iterations"] = "1500"
# params.pop("mo_preferences", None)

# labels = ["130", "134", "135", "136"]  # "132", "133",
all_labels = ["132", "133", "134", "136", "target"]
# bilabels = [(mo_label, target) for i, mo_label in enumerate(all_labels[:-1]) for target in all_labels[i+1:]]
bilabels = [('132', '133'), ('132', 'target'), ('134', '136'), ('134', 'target')]
features = {"132", "133", "134", "136"}
logs = dict()
scriptlines = {comb: [] for comb in mg_combinators}

# --------- Get Baseline costs for all labels ---------------
baseline_folder = "baseline_results"
baseline_log_results_file = os.path.join(baseline_folder, "log_results.json")
with open(baseline_log_results_file, 'r') as f:
    baseline_log_results = json.load(f)
# Get baseline costs
last_iter_baseline = 600
baseline_costs = dict()
for i, label in enumerate(all_labels):
    logfile = os.path.join(baseline_folder, baseline_log_results[label])
    results = log2dataframe(logfile, ["Training-loss"])["Training-loss"]
    results.columns = all_labels
    baseline_costs[label] = results.iloc[-1]
# ------------------------

nub = 5
for (mo_label, target) in bilabels:
    params["label_column"] = "name:" + target
    params["mo_labels"] = "name:" + mo_label
    if target != "target":
        # params["ignore_column"] = "name:target"
        params["ignore_column"] = "name:target," + ",".join(list(features - {target}))
    else:
        # params.pop("ignore_column", None)
        params["ignore_column"] = "name:" + ",".join(list(features))
    ubs = np.linspace(baseline_costs[mo_label][mo_label], baseline_costs[target][mo_label], nub)
    logs[(mo_label, target)] = []
    for ub in ubs:
        ub_data = {"ub": ub}
        for combinator in mg_combinators:
            params["mg_combination"] = combinator
            params["mo_ub_sec_obj"] = ",".join([str(ub)])
            # todo: change the namings of configfile and logfile
            configfilename = f"{combinator}_{mo_label}-{target}_" + "{:.3}.conf".format(ub)
            configfile = os.path.join(outfolder, configfilename)
            dict2config(params, configfile)

            logfile = f"{combinator}_{mo_label}-{target}_" + "{:.3}.log".format(ub)
            ub_data[combinator] = logfile
            scriptline = lightgbm + " config=" + configfilename + " 2>&1 | tee " + logfile
            scriptlines[combinator].append(scriptline)
        logs[(mo_label, target)].append(ub_data)

for comb in mg_combinators:
    with open(runscripts[comb], 'w') as f:
        f.write('\n'.join(scriptlines[comb]))

with open(log_results, 'wb') as f:
    pkl.dump(logs, f)
