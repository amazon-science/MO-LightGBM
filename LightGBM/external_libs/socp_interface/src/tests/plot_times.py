import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')

labels = [
    "100\n5",
    "300\n10",
    "500\n20",
    "1000\n30",
    "2000\n40",
    "4000\n50",
    "7500\n60",
    "10000\n70",
    "20000\n80"
]

ecos_times = np.array([
0.0011734157724999973,
0.005668423960400008,
0.01608764523600002,
0.04841508244000001,
0.14198140313999996,
0.45146086982999983,
1.56608943896,
3.17385711592,
11.7666148967,
])

eicos_times = np.array([
0.0013184142530999982,
0.006242573156400012,
0.01644091752700001,
0.05168814468999998,
0.15213851564,
0.4482394814400001,
1.2871749409799997,
2.7549623923999995,
10.5422071726,
])

plt.yscale('log')
plt.plot(labels, ecos_times, ".", linestyle="-", label="ECOS")
plt.plot(labels, eicos_times, ".", linestyle="-", label="EiCOS")

plt.xlabel('assets/factors')
plt.ylabel('avg. solve time [s]')
plt.legend()

plt.show()
