# %%

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from sklearn.utils.extmath import weighted_mode

# %% DATA AND WEIGHT
path = "ensemble"
results = pd.read_csv("results.csv")
submissions = pd.read_csv("submissions.csv")

k = 11
n7 = 7935
s = submissions.values
m, n = s.shape

# %% COUNTS
count7_true = np.array([680,790,680,700,680,700,750,680,700,770,770])
count8_true = np.array([710,800,750,780,720,720,780,780,780,780,790])

count7_pred, count8_pred = [], []
for i in range(n):
    count7, bin_edges = np.histogram(s[:n7,i], bins=range(1,13))
    count8, bin_edges = np.histogram(s[n7:,i], bins=range(1,13))
    count7_pred.append(count7)
    count8_pred.append(count8)
count7_pred = np.stack(count7_pred)
count8_pred = np.stack(count8_pred)

error7 = np.sum(np.abs(count7_pred - count7_true), axis=1)
error8 = np.sum(np.abs(count8_pred - count8_true), axis=1)
error = error7 + error8

# %% WEIGHT
# more weight, more importance
w1 = results["test_acc"].values
w1 = w1 / np.max(w1)
w2 = results["epoch"].values
w2 = w2 / np.max(w2)
w3 = results["test_size"].values
w3 = w3 / np.max(w3)
w4 = 1/error**2
w4 = w4 / np.max(w4)

a1, a2, a3, a4 = 0,0,0,1
w = a1*w1 + a2*w2 + a3*w3 + a4*w4

# w = np.array([
# 0,0,5,0,5,0,0,5,0,1,0,1,
# 1,10,5,10,100,10,0,1,10,0,0,0,
# 0,0,0,3,0,0,0,1,0,0,0,0,
# 0,0,5,0,5,5,1,0,5,1,0,0,
# 5,0,1,3,0,3,10,0,1,5,5,0,
# 1,10,100,10,0,3,5,3,0,0,0,0,
# 0,5,0,3,5,3,0,5,5,0,10,0,
# 0,1,5,0,10,5,10,0,0,1,0,0,
# 5,0,1,0,3,3,0,5,0,0,0,0,
# 0,0,0,0,0,0,0,0,1,0,0,1,
# 0,0
# ])
# w = np.ones_like(w)

remove_baseline = True
if remove_baseline:
    # WITHOUT BASELINE MODEL
    s = s[:,9:]
    w = w[9:]
m, n = s.shape

if False:
    # %% MODELS HISTOGRAM
    width = int(np.sqrt(n))
    height = width+1

    fig, axs = plt.subplots(width, height, sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0, hspace=0)
    axs = axs.flatten()
    for i in range(n):
        axs[i].hist(s[:n7,i], bins=range(0,13))
    plt.savefig(os.path.join(path, "models_hist_7.pdf"))

    fig, axs = plt.subplots(width, height, sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0, hspace=0)
    axs = axs.flatten()
    for i in range(n):
        axs[i].hist(s[n7:,i], bins=range(0,13))
    plt.savefig(os.path.join(path, "models_hist_8.pdf"))


# %% COUNTS

counts, orders, counts_ordered = [], [], []
for i in s:
    count = np.bincount(i, minlength=k+1)
    order = count.argsort()[::-1]
    counts.append(count)
    orders.append(order)
    counts_ordered.append(count[order])
counts = np.vstack(counts)
orders = np.vstack(orders)
counts_ordered = np.vstack(counts_ordered)
counts_ordered = counts_ordered[counts_ordered[:, 0].argsort()]

fig, ax = plt.subplots()
g = ax.imshow(counts_ordered, resample=False, interpolation='none')
plt.colorbar(g)
ax.set_aspect(1./ax.get_data_ratio())
ax.set_xlabel("Ordered label")
ax.set_ylabel("Instance")
plt.tight_layout()
plt.savefig(os.path.join(path, "count_heatmap.pdf"))

fig, axs = plt.subplots(3,4, sharex=True, sharey=True)
plt.subplots_adjust(wspace=0, hspace=0)
axs = axs.flatten()
for i in range(1,12):
    axs[i].imshow(counts_ordered[orders[:,0]==i], resample=False, interpolation='none')
    axs[i].set_aspect(1./axs[i].get_data_ratio())
    # axs[i].set_xlabel("Ordered label")
    # axs[i].set_ylabel("Instance")
    axs[i].set_title("class " + str(i))
plt.tight_layout()
plt.savefig(os.path.join(path, "count_heatmap_class.pdf"))


if False:
    fig, ax = plt.subplots()
    ax.scatter(counts_ordered[:,0], counts_ordered[:,1], alpha=0.1, s=1)
    ax.axvline(n//2, color="black", ls="dashed", lw=0.5)
    ax.axhline(n//2, color="black", ls="dashed", lw=0.5)
    ax.set_xlim(0,n)
    ax.set_ylim(0,n)
    ax.set_xlabel("First mode")
    ax.set_xlabel("Second mode")
    plt.tight_layout()
    plt.savefig(os.path.join(path, "mode_1vs2.pdf"))

# %% MODE STATS

# mode, count = scipy.stats.mode(s, axis=1)
mode, count = weighted_mode(s, w, axis=1)
# mode = np.average(s, axis=1, weights=w)
# mode = np.round(mode)
# mode = np.expand_dims(mode, axis=1)

count = np.sum(s == mode, axis=1)
bincount = np.bincount(count, minlength=n)

accuracy = np.cumsum(np.flip(bincount)) / m
accuracy = np.flip(accuracy)

fig, ax = plt.subplots(figsize=(16,10))

c1 = "blue"
ax.bar(np.arange(n+1), bincount, width=0.8, color=c1, alpha=0.5)
for b in range(n+1):
    ax.text(b, bincount[b], bincount[b], ha="center", va="bottom", rotation=0, fontsize=6, color=c1)
ax.set_ylabel("Mode Count", color=c1)

c2 = "red"
ax2 = ax.twinx()
ax2.plot(accuracy, color=c2)
for b in range(n+1):
    ax2.text(b, accuracy[b]*1.0, "{:.6f}".format(accuracy[b]), ha="center", va="top", rotation=90, fontsize=6, color=c2)
ax2.set_ylim(0,1.2)
ax2.set_ylabel("Accuracy", color=c2)

plt.axvline(n//2, color="black", ls="dashed", lw=0.5)

plt.tight_layout()
plt.savefig(os.path.join(path, "mode_count.pdf"))

# %%

if False:
    idx = np.where(count == 40)[0]
    for k, i in enumerate(idx):
        print(i, s[i],  np.bincount(s[i], minlength=n), int(mode[i]))
        if k > 10:
            break

# %% SAVE SUBMISSION
to_submit = s[:,w.argmax()]
to_submit = mode

fig, axs = plt.subplots(1,2, sharey=True)
axs[0].hist(to_submit[:n7], bins=np.arange(0.5,12.5), color="blue", alpha=0.5, width=1.0)
axs[0].bar(x=range(1,12), height=count7_true, color="orange", alpha=0.5, width=0.5)
axs[0].set_title("Individual 7")
axs[0].set_xticks([1,2,3,4,5,6,7,8,9,10,11])
axs[1].hist(to_submit[n7:], bins=np.arange(0.5,12.5), color="blue", alpha=0.5, width=1.0)
axs[1].bar(x=range(1,12), height=count8_true, color="orange", alpha=0.5, width=0.5)
axs[1].set_title("Individual 8")
axs[1].set_xticks([1,2,3,4,5,6,7,8,9,10,11])
axs[0].set_ylim(0, 900)

plt.savefig(os.path.join(path, "submit_hist_class.pdf"))

# %%

np.savetxt("igorgo.txt", to_submit, fmt='%d')