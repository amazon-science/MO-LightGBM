import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends import backend_pdf
import os
from latex_utils import latexify
from utils import log2dataframe
import yaml
import sys

configfile = sys.argv[1]
with open(configfile, 'r') as f:
    config = yaml.safe_load(f)

plot_iter = None
#print(len(sys.argv))
if len(sys.argv)==3:
    plot_iter = int(sys.argv[2])

dataset = config["dataset"]
objective_type = config["lightgbm_parameters"]["objective_type"]

# Relevant paths
config_root = os.path.dirname(os.path.abspath(configfile))
objective_type_root = os.path.join(config_root
                                   , f"results_{dataset['name']}"
                                   , f"{objective_type}"
                                   )

folder_name = "figures"
if "folder_name" in config["plotting"]:
    folder_name = config["plotting"]["folder_name"]

outfolder = os.path.join(objective_type_root
                         , "biobjective_results"
                         , folder_name
                         )

if not os.path.isdir(outfolder):
    os.makedirs(outfolder)

baseline_folder = os.path.join(objective_type_root, f"baseline_results")
biobjective_folder = os.path.join(objective_type_root, f"biobjective_results")
with open(os.path.join(baseline_folder, "log_results.yml"), 'r') as f:
    baseline_log_results = yaml.safe_load(f)
with open(os.path.join(biobjective_folder, "log_results.yml"), 'r') as f:
    pref_biobjective_log_results = yaml.load(f, Loader=yaml.FullLoader)
with open(os.path.join(biobjective_folder, "ec_log_results.yml"), 'r') as f:
    ec_biobjective_log_results = yaml.load(f, Loader=yaml.FullLoader)

pref_combinators = config['mg_combinators']['preference_based']
ec_combinators = config['mg_combinators']['constraint_based']
mg_combinators = pref_combinators + ec_combinators

comb_legends = config['plotting']['combinator_legends']
markers = config['plotting']['combinator_markers']
msz = config['plotting']['combinator_marker_sizes']

all_labels = dataset['all_labels']
bilabels_idx = dataset['bilabels_idx']
bilabels = [(all_labels[i], all_labels[j]) for i, j in bilabels_idx]
label_legends = config['plotting']['label_legends']

num_prefs = config['num_tradeoffs']
var_types = config['plotting']['to_track']
variables = list(var_types.values())
#last_iter = int(config['lightgbm_parameters']['num_iterations'])
#if plot_iter:
#    last_iter = plot_iter

# snapshots_at = np.arange(0, last_iter)  # int(last_iter*0.01)
snapshots_at = config['plotting']['snapshots_at']
if plot_iter:
    snapshots_at = [plot_iter]
else:
    snapshots_at = [snapshots_at[-1]]
#print("last iter = ", last_iter)
print("snapshots_at = ", snapshots_at)
# remove_iters = [] # [1,2]
# snapshots_at = np.delete(snapshots_at, remove_iters)
# snapshots_at = [10, 20, 40, 80, 160, 240, 480, 600]

def get_bilabel_data(log_results, combinators, parameter='preferences'):
    data = dict()
    for label_pair in bilabels:
        data[label_pair] = {combinator: {"loss": [], "ndcg": []} for combinator in combinators}
        data[label_pair][parameter] = []
        for i, param_data in enumerate(log_results[label_pair]):
            data[label_pair][parameter].append(param_data[parameter])
            for combinator in combinators:
                logfilename = param_data[combinator]
                logfile = os.path.join(biobjective_folder, combinator, logfilename)
                results = log2dataframe(logfile, variables)
                for vname, vtype in var_types.items():
                    try:
                        results_at_snapshots = results[vtype].iloc[snapshots_at].to_numpy().T
                    except IndexError as e:
                        ealy_stop_iter = results[vtype].shape[0]
                        if ealy_stop_iter < 1:
                            raise IndexError(e)
                        print(f'**Warning**: {logfilename} early stopped at {ealy_stop_iter}')
                        temp_snapshots_at = [ss for ss in snapshots_at if ss < ealy_stop_iter-1] + [ealy_stop_iter-1]
                        temp_result = results[vtype].iloc[temp_snapshots_at].to_numpy().T
                        last_results = [temp_result[:, -1] for _ in range(len(snapshots_at) - len(temp_snapshots_at))]
                        #results_at_snapshots = np.column_stack([temp_result] + last_results)
                        results_at_snapshots = temp_result
                    data[label_pair][combinator][vname].append(results_at_snapshots)

        # make everything ndarray
        data[label_pair][parameter] = np.asarray(data[label_pair][parameter])
        for combinator in combinators:
            for vname in var_types:
                data[label_pair][combinator][vname] = np.asarray(data[label_pair][combinator][vname])

    return data


pref_data = get_bilabel_data(pref_biobjective_log_results, pref_combinators, parameter='preferences')
ec_data = get_bilabel_data(ec_biobjective_log_results, ec_combinators, parameter='ub')

# -------------- Get baseline data -----------------
baseline_results = dict()
baseline_all = dict()
for label in all_labels:
    logfile = os.path.join(baseline_folder, baseline_log_results[label])
    results = log2dataframe(logfile, variables)
    baseline_results[label] = dict()
    baseline_all[label] = dict()
    for vname, vtype in var_types.items():
        results[vtype].columns = all_labels
#        print(results[vtype])
        early_stop_iter = results[vtype].shape[0]-1
        _last_iter = min(early_stop_iter, snapshots_at[-1])
#        print(f"lastiter = {_last_iter}")
        baseline_results[label][vname] = results[vtype].iloc[_last_iter][label]  # .iloc[snapshots_at][0].to_numpy()
        baseline_all[label][vname] = results[vtype].iloc[_last_iter]
#        baseline_results[label][vname] = results[vtype].iloc[snapshots_at[-1]][label]  # .iloc[snapshots_at][0].to_numpy()
#        baseline_all[label][vname] = results[vtype].iloc[snapshots_at[-1]]
# plot data for baseline
baseline_data = dict()
baseline_ulimit = dict()
for label_pair in bilabels:
    sec_label, prim_label = label_pair
    baseline_data[label_pair] = dict()
    baseline_ulimit[label_pair] = dict()
    for vname in var_types:
        baseline_data[label_pair][vname] = (baseline_results[sec_label][vname], baseline_results[prim_label][vname])
        baseline_ulimit[label_pair][vname] = [baseline_all[prim_label][vname][sec_label]
                                            , baseline_all[sec_label][vname][prim_label]]
# --------------------------------------------------
#print(baseline_ulimit)

def scatter_plot_results(combinators, data, variable, ax, legends=None):
    outDF = pd.DataFrame()
    for combinator in combinators:
        res = data[combinator][variable]
        lgnd = '' if legends is None else legends[combinator]
        ax.scatter(res[:, 0, -1], res[:, 1, -1], marker=markers[combinator], edgecolor='k', lw=0.3,
                   c=colors, s=msz[combinator], label=lgnd, zorder=10)
        a = pd.DataFrame(res[:,[0,1],-1])
        a.columns = ['obj0','obj1']
        a.insert(0, 'pref', a.index.to_numpy())
        a.insert(0, 'combinator', combinator)
        a = a.pivot(index='combinator', columns='pref', values=['obj0','obj1']).swaplevel(0,1,1).sort_index(axis=1)
        a.reset_index(inplace=True)

        if outDF.empty:
            outDF = a
        else:
            outDF = pd.concat([outDF, a], axis=0)
#    print(outDF)
    return outDF


colors = sns.color_palette("tab10", num_prefs)
pdfs = {'pref': backend_pdf.PdfPages(f"{outfolder}/{dataset['name']}-pref-biobjectives-{snapshots_at[-1]}.pdf"),
        'ec': backend_pdf.PdfPages(f"{outfolder}/{dataset['name']}-ec_biobjectives-{snapshots_at[-1]}.pdf"),
        'all': backend_pdf.PdfPages(f"{outfolder}/{dataset['name']}-biobjectives-{snapshots_at[-1]}.pdf")
        }

plotdata_file = {'loss': f"{outfolder}/{dataset['name']}-loss-{snapshots_at[-1]}.tsv"
                 , 'ndcg': f"{outfolder}/{dataset['name']}-ndcg-{snapshots_at[-1]}.tsv"
                 }

latexify(fig_width=4, fig_height=1.8)
loss_plotdata_all = pd.DataFrame()
ndcg_plotdata_all = pd.DataFrame()

for pi, label_pair in enumerate(bilabels):
    mo_label, target = label_pair
    figs = {'pref': plt.figure(), 'ec': plt.figure(), 'all': plt.figure()}
    pref_fig, ec_fig, all_fig = plt.figure(), plt.figure(), plt.figure()
    axs = {'loss': {'pref': figs['pref'].add_subplot(121), 'ec': figs['ec'].add_subplot(121), 'all': figs['all'].add_subplot(121)},
           'ndcg': {'pref': figs['pref'].add_subplot(122), 'ec': figs['ec'].add_subplot(122), 'all': figs['all'].add_subplot(122)}
           }
    
    # ------------------ LOSS PLOT -------------------
    # axis limits
    loss_ulim = np.max(np.stack([np.max(pref_data[label_pair][comb]['loss'], axis=(0, 2)) for comb in pref_combinators]), axis=0)
    loss_ulim_ec = np.max(np.stack([np.max(ec_data[label_pair][comb]['loss'], axis=(0, 2)) for comb in ec_combinators]), axis=0)
    loss_ulim = np.max(np.stack([loss_ulim, loss_ulim_ec], axis=0), axis=0)
    # for paper plot
    loss_ulim = np.min([loss_ulim, baseline_ulimit[label_pair]['loss']], axis=0)
    sec_loss, prim_loss = baseline_data[label_pair]["loss"]   # baseline losses
    for ax in axs['loss'].values():
        ax.set_xlim(-loss_ulim[0]*0.05, loss_ulim[0]*1.05)
        ax.set_ylim(-loss_ulim[1]*0.05, loss_ulim[1]*1.05)
        ax.set_title(config['plotting']['tracker_titles']['loss'])
        ax.set_xlabel(f"{label_legends[mo_label]}", labelpad=1)
        ax.set_ylabel(f"{label_legends[target]}")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Plot baseline
        ax.plot([0, loss_ulim.sum()], [prim_loss, prim_loss], lw=2, alpha=0.4, c='k')   # horizontal line
        ax.plot([sec_loss, sec_loss], [0, loss_ulim.sum()], lw=2, alpha=0.4, c='k')     # vertical line

    # plot preference rays and trajectories
    for pref_id, r in enumerate(pref_data[label_pair]["preferences"]):
        lgnd = r"$\mathbf{r}^{-1}$ Ray" if pref_id == 0 and pi in [0] else ""
        r_inv = 1. / r
        r_inv /= np.linalg.norm(r_inv)
        for ax in ['pref', 'all']:
            axs['loss'][ax].plot([0, loss_ulim.sum() * r_inv[0]], [0, loss_ulim.sum() * r_inv[1]],
                                     color=colors[pref_id], lw=0.5, alpha=0.5, ls='--', dashes=(10, 2), label=lgnd)

    # plot ub bound lines
    for ub_id, ub in enumerate(ec_data[label_pair]['ub']):
        lgnd = r"$\epsilon$ bound" if ub_id == 0 and pi in [0] else ""
        for ax in ['ec', 'all']:
            axs['loss'][ax].plot([ub, ub], [0, loss_ulim.sum()], color=colors[ub_id],
                                   lw=0.5, alpha=0.5, ls='-', label=lgnd)
    # Plot end points
#    print(pref_combinators)
#    print(pref_data[label_pair])
#    exit(1)
    pref_plotdata = scatter_plot_results(pref_combinators, pref_data[label_pair], 'loss', axs['loss']['pref'])
    ec_plotdata = scatter_plot_results(ec_combinators, ec_data[label_pair], 'loss', axs['loss']['ec'])
    loss_plotdata = pd.concat([pref_plotdata, ec_plotdata], axis=0)
    for i in reversed(range(len(label_pair))):
        loss_plotdata.insert(0, f'label_{i}', label_pair[i])
    if loss_plotdata_all.empty:
        loss_plotdata_all = loss_plotdata
    else:
        loss_plotdata_all = pd.concat([loss_plotdata_all, loss_plotdata], axis=0)


    scatter_plot_results(pref_combinators, pref_data[label_pair], 'loss', axs['loss']['all'])
    scatter_plot_results(ec_combinators, ec_data[label_pair], 'loss', axs['loss']['all'])

    if pi in [0]:
        for ax in axs['loss'].values():
            ax.legend(fontsize=7, loc='upper right', facecolor=(1, 1, 1, 1))

    # # -------------- NDGC plot ---------------
    llim = np.min(np.stack([np.min(pref_data[label_pair][comb]['ndcg'], axis=(0, 2)) for comb in pref_combinators]), axis=0)
    ulim = np.max(np.stack([np.max(pref_data[label_pair][comb]['ndcg'], axis=(0, 2)) for comb in pref_combinators]), axis=0)
    sec_ndcg, prim_ndcg = baseline_data[label_pair]["ndcg"]
    for trade_off_type, ax in axs['ndcg'].items():
        ax.set_title(config['plotting']['tracker_titles']['ndcg'], y=-0.2)
        ax.set_xlabel(f"{label_legends[mo_label]}")
        ax.set_ylabel(f"{label_legends[target]}")
        x_range = ulim[0]-llim[0]
        y_range = ulim[1]-llim[1]
        ax.set_xlim(llim[0] - x_range * 0.05, llim[0] + x_range * 1.05)
        ax.set_ylim(llim[1] - y_range * 0.05, llim[1] + y_range * 1.05)
        # Hide the right and top spines
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('right')
        ax.xaxis.set_ticks_position('top')
        ax.yaxis.set_label_position('right')
        ax.xaxis.set_label_position('top')

        # Plot baseline
        lgnd = "baseline" if pi == 0 and trade_off_type == 'ec' else ""
        ax.plot([0, 1], [prim_ndcg, prim_ndcg], lw=2, alpha=0.4, c='k', label=lgnd)   # horizontal line
        ax.plot([sec_ndcg, sec_ndcg], [0, 1], lw=2, alpha=0.4, c='k')                 # vertical line

    legends = comb_legends if pi in [0, 2] else None
    pref_plotdata = scatter_plot_results(pref_combinators, pref_data[label_pair], 'ndcg', axs['ndcg']['pref'], legends=legends)
    ec_plotdata = scatter_plot_results(ec_combinators, ec_data[label_pair], 'ndcg', axs['ndcg']['ec'], legends=legends)
    ndcg_plotdata = pd.concat([pref_plotdata, ec_plotdata], axis=0)
    for i in reversed(range(len(label_pair))):
        ndcg_plotdata.insert(0, f'label_{i}', label_pair[i])
    if ndcg_plotdata_all.empty:
        ndcg_plotdata_all = ndcg_plotdata
    else:
        ndcg_plotdata_all = pd.concat([ndcg_plotdata_all, ndcg_plotdata], axis=0)

    scatter_plot_results(pref_combinators, pref_data[label_pair], 'ndcg', axs['ndcg']['all'], legends=legends)
    scatter_plot_results(ec_combinators, ec_data[label_pair], 'ndcg', axs['ndcg']['all'], legends=legends)

    if pi in [0]:
        for ax in axs['ndcg'].values():
            ax.legend(fontsize=7, loc='lower left', facecolor=(1, 1, 1, 1))
    for trade_off_type in pdfs:
        pdf, fig = pdfs[trade_off_type], figs[trade_off_type]
        fig.tight_layout()
        fig.subplots_adjust(left=0.12, right=0.87, top=0.78 if pi > 1 else 0.82, bottom=0.175)
        pdf.savefig(fig)
        plt.close(fig)

for pdf in pdfs.values():
    pdf.close()

loss_plotdata_all.to_csv(plotdata_file['loss'], sep='\t', index=False)
ndcg_plotdata_all.to_csv(plotdata_file['ndcg'], sep='\t', index=False)
