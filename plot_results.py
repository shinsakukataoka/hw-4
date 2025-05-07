import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sizes      = [2 ** i for i in range(1, 11)]     # 2 .. 1024
labels_x   = list(range(1, 11))                  # exponent for x‑axis

v1avg = []; v2avg = []; vnavg = []
v1err = []; v2err = []; vnerr = []

for n in sizes:
    fname = Path(f"{n}.out")
    if not fname.exists():
        print(f"warning: {fname} missing – skipped")
        continue
    d = np.loadtxt(fname)
    if d.ndim == 1:
        d = d.reshape(1, -1)    # handle 1‑line case

    # mean
    v1avg.append(np.mean(d[:, 0])); v2avg.append(np.mean(d[:, 1])); vnavg.append(np.mean(d[:, 2]))
    # 95 % confidence interval
    v1err.append(1.96 * np.std(d[:, 0], ddof=1) / np.sqrt(len(d[:, 0])))
    v2err.append(1.96 * np.std(d[:, 1], ddof=1) / np.sqrt(len(d[:, 1])))
    vnerr.append(1.96 * np.std(d[:, 2], ddof=1) / np.sqrt(len(d[:, 2])))

plt.errorbar(labels_x, v1avg, yerr=v1err, lw=0.7, elinewidth=0.7, capsize=2, markersize=5, label='One thread')
plt.errorbar(labels_x, v2avg, yerr=v2err, lw=0.7, elinewidth=0.7, capsize=2, markersize=5, label='Two threads')
plt.errorbar(labels_x, vnavg, yerr=vnerr, lw=0.7, elinewidth=0.7, capsize=2, markersize=5, label='N threads')

plt.legend(loc='best')
plt.xlabel('N = 2^x')
plt.ylabel('Turnaround time (ns)')
plt.tight_layout()
plt.savefig("timings.png", dpi=300)
