import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sizes    = [2 ** i for i in range(1, 11)]         # 2 … 1024
exp_idx  = list(range(1, 11))                      # x‑axis = exponent

v1avg=v2avg=vnavg=[]
v1err=v2err=vnerr=[]

v1avg=[]; v2avg=[]; vnavg=[]
v1err=[]; v2err=[]; vnerr=[]

for n in sizes:
    p = Path(f"{n}.out")
    d = np.loadtxt(p)
    if d.ndim == 1:
        d = d.reshape(1, -1)

    d = d / 1_000.0   # ns → μs

    v1avg.append(np.mean(d[:,0])); v2avg.append(np.mean(d[:,1])); vnavg.append(np.mean(d[:,2]))
    v1err.append(1.96*np.std(d[:,0],ddof=1)/np.sqrt(len(d)))
    v2err.append(1.96*np.std(d[:,1],ddof=1)/np.sqrt(len(d)))
    vnerr.append(1.96*np.std(d[:,2],ddof=1)/np.sqrt(len(d)))

def make_plot(yscale:str, outfile:str):
    plt.figure(figsize=(7,5))
    plt.errorbar(exp_idx, v1avg, yerr=v1err, lw=0.7, elinewidth=0.7, capsize=2, markersize=5, label='One thread')
    plt.errorbar(exp_idx, v2avg, yerr=v2err, lw=0.7, elinewidth=0.7, capsize=2, markersize=5, label='Two threads')
    plt.errorbar(exp_idx, vnavg, yerr=vnerr, lw=0.7, elinewidth=0.7, capsize=2, markersize=5, label='N threads')
    plt.yscale(yscale)
    plt.xlabel('N = 2^x')
    plt.ylabel('Turnaround time (μs)')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"saved → {outfile}")

make_plot('linear', 'timings_us_linear.png')
make_plot('log',    'timings_us_log.png')

