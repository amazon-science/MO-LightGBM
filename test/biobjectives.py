import os
from utils import config2dict, dict2config, circle_points, log2dataframe
import numpy as np
import pickle as pkl
import json

mg_combinators = ["linear_scalarization",
                  # "stochastic_label_aggregation",
                  "chebychev_scalarization",
                  "epo_search",
                  # "wc_mgda",
                 ]

# Relevant paths
cwd = os.getcwd()
outfolder = os.path.join(cwd, "biobjective_results")
runscripts = {comb: os.path.join(outfolder, comb + "_run_experiment.sh") for comb in mg_combinators}
lightgbm = os.path.join(cwd, os.pardir, "LightGBM", "lightgbm")
log_results = os.path.join(outfolder, "log_results.pkl")
sampleconfig = os.path.join(cwd, "sample_config.conf")
trainfile = os.path.join(cwd, "datasets", "inverted_qs_train.tsv")
validfile = os.path.join(cwd, "datasets", "inverted_qs_valid.tsv")

# Read sample config and make changes for single objective ranking optimization
params = config2dict(sampleconfig)
params["train_data_file"] = trainfile
params["valid_data_file"] = validfile
npref = 5
# preferences = circle_points(npref, min_angle=0.0001*np.pi/2, max_angle=0.9999*np.pi/2)

# labels = ["130", "134", "135", "136"]  # "132", "133",
all_labels = ["132", "133", "134", "136", "target"]
# bilabels = [(mo_label, target) for i, mo_label in enumerate(all_labels[:-1]) for target in all_labels[i+1:]]
bilabels = [('132', '133'), ('132', 'target'), ('134', '136'), ('134', 'target')]
features = {"132", "133", "134", "136"}
logs = dict()
scriptlines = {comb: [] for comb in mg_combinators}

# Get baseline data
baseline_folder = "baseline_results"
baseline_log_results_file = os.path.join(baseline_folder, "log_results.json")
with open(baseline_log_results_file, 'r') as f:
    baseline_log_results = json.load(f)
baseline_data = dict()
for i, label in enumerate(all_labels):
    logfile = os.path.join(baseline_folder, baseline_log_results[label])
    results = log2dataframe(logfile, ["Training-loss"])["Training-loss"]
    results.columns = all_labels
    baseline_data[label] = results.iloc[-1]


for (mo_label, target) in bilabels:
    params["label_column"] = "name:" + target
    params["mo_labels"] = "name:" + mo_label
    if target != "target":
        # params["ignore_column"] = "name:target"
        params["ignore_column"] = "name:target," + ",".join(list(features - {target}))
    else:
        # params.pop("ignore_column", None)
        params["ignore_column"] = "name:" + ",".join(list(features))
    logs[(mo_label, target)] = []
    min_angle = np.arctan(baseline_data[mo_label][mo_label] / baseline_data[mo_label][target])
    max_angle = np.arctan(baseline_data[target][mo_label] / baseline_data[target][target])
    preferences = circle_points(npref, min_angle=min_angle, max_angle=max_angle)
    
    for pref in preferences:
        pref_data = {"preferences": pref}
        for combinator in mg_combinators:
            params["mg_combination"] = combinator
            params["mo_preferences"] = ",".join([str(p) for p in pref])
            # todo: change the namings of configfile and logfile
            configfilename = f"{combinator}_{mo_label}-{target}_" + "{:.3}-{:.3}.conf".format(pref[0], pref[1])
            configfile = os.path.join(outfolder, configfilename)
            dict2config(params, configfile)

            logfile = f"{combinator}_{mo_label}-{target}_" + "{:.3}-{:.3}.log".format(pref[0], pref[1])
            pref_data[combinator] = logfile
            scriptline = lightgbm + " config=" + configfilename + " 2>&1 | tee " + logfile
            scriptlines[combinator].append(scriptline)
        logs[(mo_label, target)].append(pref_data)

for comb in mg_combinators:
    with open(runscripts[comb], 'w') as f:
        f.write('\n'.join(scriptlines[comb]))
        
with open(os.path.join(outfolder, "run_experiment.sh"), 'w') as f:
    f.write('\n'.join(['\n'.join(scriptlines[comb]) for comb in mg_combinators]))

with open(log_results, 'wb') as f:
    pkl.dump(logs, f)
