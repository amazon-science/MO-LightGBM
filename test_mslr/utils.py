import os
import pandas as pd
import re
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba


def varying_alpha_line(x, y, ax=None, c='blue', lw=1):
    if ax is None:
        ax = plt.gca()
    mid_alpha = 0.25
    for i in range(len(x)-1):
        a = (i+1)/len(x)
        a = (mid_alpha-0.05)*a/0.75 + 0.05 if a < 0.75 else (1-0.25)*(a-0.75)/(1-0.75) + mid_alpha
        ax.plot(x[i:i+2],y[i:i+2], lw=lw, color=to_rgba(c, a))


def config2dict(configfile):
    config = open(configfile, "r")
    params = dict()
    for line in config:
        if line.strip().startswith("#") or len(line.strip()) == 0:
            continue
        try:
            param, value = [txt.strip() for txt in line.split("=")]
        except ValueError as e:
            print([txt.strip() for txt in line.split("=")])
            raise ValueError(e)
        params[param] = value
    config.close()
    return params


def dict2config(params, configfile):
    lines = []
    for param, value in params.items():
        lines.append(" = ".join([param, value]))
    with open(configfile, "w") as config:
        config.write("\n".join(lines) + '\n')


def sample_from_simplex(d, n, M=100, normalize=False):
    """
    Samples d-dimensional vectors distributed uniformly on the (d-1)-simplex
    number of total discrete points on simplex from which samples are drawn = comb(M-1, d) * d
    refer https://www.cs.cmu.edu/~nasmith/papers/smith+tromble.tr04.pdf
    :param d: int, dimension of the vector
    :param n: int, number of samples
    :return: list of lists
    """
    samples = []
    for _ in range(n):
        x = random.sample(range(1, M), d-1)
        x = [0] + sorted(x) + [M]
        y = [(x[i+1] - x[i]) / M if normalize else x[i+1] - x[i] for i in range(d)]
        samples.append(y)
    return samples


def sample_inverse_preferences(data, num_pref):
    n, d = data.shape
    preferences = list(data)
    stack = [data]
    while len(preferences) < num_pref:
        corners = stack.pop(0)
        new_pref = corners.sum(axis=0) / n
        preferences.append(new_pref)
        for i in range(n):
            new_corners = corners.copy()
            new_corners[i] = new_pref
            stack.append(new_corners)

    return preferences



def log2dataframe(logfile, variables):
    log = open(logfile, "r")
    results = {variable: [] for variable in variables}
    for line in log:
        for variable in variables:
            if variable not in line:
                continue
            values_str = re.search(r"\[[0-9.,\s]+\]", line).group()
            values = [float(value) for value in values_str[1:-1].split(',') if len(value) > 0]
            results[variable].append(values)
    log.close()
    for variable in variables:
        results[variable] = pd.DataFrame(results[variable])
    return results


def circle_points(K, min_angle=None, max_angle=None):
    # generate evenly distributed preference vector
    ang0 = np.pi / 20. if min_angle is None else min_angle
    ang1 = np.pi * 9 / 20. if max_angle is None else max_angle
    angles = np.linspace(ang0, ang1, K)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.c_[x, y]


if __name__ == "__main__":
    logfolder = "../LTRdatasets"
    logname = "log_epo.txt"
    logfile = os.path.join(logfolder, logname)
    variables = ["Preferences",
                 "Training-ndcg@5", "Validation-ndcg@5",
                 "Training-ndcg@30", "Validation-ndcg@30",
                 "Training-loss", "Validation-loss", "combination-coefficients"]
    results = log2dataframe(logfile, variables)
    for var in variables:
        print(var, ":", results[var].shape)
