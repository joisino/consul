import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import pairwise_distances
import csv
import subprocess
from scipy.optimize import minimize
import networkx as nx


plt.rcParams['font.family'] = 'Segoe UI'
plt.rcParams['pdf.fonttype'] = 42

a = []
n = 100
K = 9

with open('adult.data') as f:
    reader = csv.reader(f)
    for r in reader:
        if len(r) == 15 and int(r[0]) < 90 and 0 < int(r[10]) and int(r[10]) < 99999:
            a.append([int(r[0]), np.log10(int(r[10]))])

a = np.array(a)[:n]
mu = np.mean(a, axis=0, keepdims=True)
std = np.std(a, axis=0, keepdims=True)
a = (a - mu) / std

D = pairwise_distances(a)

with open('adult_2dim_A.txt', 'w') as f:
    for i in range(n):
        ss = set(np.argsort(D[i])[:K + 1].tolist())
        for j in range(n):
            if j in ss:
                print(1, end=' ', file=f)
            else:
                print(0, end=' ', file=f)
        print(file=f)

subprocess.call('docker run -ti --rm -v "$PWD":/home/docker -w /home/docker --net host r-base Rscript adult.R', shell=True)

with open('adult_2dim_LOE.txt') as f:
    a_hat = np.array(list(map(float, f.read().split()))).reshape(2, n).T
    a_hat = (a_hat - np.mean(a_hat, axis=0, keepdims=True)) / np.std(a_hat, axis=0, keepdims=True)


def convert(a_orig, mat, bias):
    a_orig = a_orig @ mat.reshape(2, 2)
    a_orig = a_orig + bias
    return a_orig


def convert_x(a_orig, x):
    return convert(a_orig, x[:4], x[4:])


def f(x):
    return (np.abs(convert_x(a_hat, x) - a) ** 2).sum()


res = minimize(f, np.zeros(6))
print(res)

a_hat = convert_x(a_hat, res.x)

a = a * std + mu
a_hat = a_hat * std + mu


anchor = [1, 2, 3, 4, 5]

fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(1, 3, 1)
for i in anchor:
    ax.text(a[i, 0] + 0.1, a[i, 1] + 0.02, str(i), fontsize=16)
ax.scatter(a[:, 0], a[:, 1], c='#005aff', s=10)
ax.scatter(a[anchor, 0], a[anchor, 1], c='#ff4b00', zorder=10)
ax.set_xticks([20, 40, 60])
ax.set_yticks([3, np.log10(2 * 10**3), np.log10(5 * 10**3), 4, np.log10(2 * 10**4)])
ax.set_yticklabels([10**3, 2 * 10**3, 5 * 10**3, 10**4, 2 * 10**4])
ax.set_xlabel('Age', fontsize=14)
ax.set_ylabel('Capital-gain', fontsize=14)
fig.text(0.23, -0.12, '(a) Confidential Features\n(not observable)', fontsize=14, horizontalalignment='center')

G = nx.DiGraph()
for i in range(n):
    ss = set(np.argsort(D[i])[:K + 1].tolist())
    for j in range(n):
        if j in ss and j != i:
            G.add_edge(i, j)

ax = fig.add_subplot(1, 3, 2)
node_color = ['#005aff' for i in range(n)]
for i, v in enumerate(G.nodes()):
    if v in anchor:
        node_color[i] = '#ff4b00'
pos = nx.spring_layout(G, k=0.4, seed=0)
# pos = nx.kamada_kawai_layout(G)
labels = {i: (str(i) if i in anchor else '') for i in range(n)}
nx.draw_networkx(G, ax=ax, pos=pos, node_size=20, labels=labels, node_color=node_color, edge_color='#84919e', width=0.2)
ax.set_position([0.375, 0.11, 0.25, 0.77])
fig.text(0.5, -0.12, '(b) Recommendation Network\n(observable)', fontsize=14, horizontalalignment='center')

ax = fig.add_subplot(1, 3, 3)
for i in anchor:
    ax.text(a_hat[i, 0] + 0.1, a_hat[i, 1] + 0.02, str(i), fontsize=16)
ax.scatter(a_hat[:, 0], a_hat[:, 1], c='#005aff', s=10)
ax.scatter(a_hat[anchor, 0], a_hat[anchor, 1], c='#ff4b00', zorder=10)
ax.set_xticks([20, 40, 60])
ax.set_yticks([3, np.log10(2 * 10**3), np.log10(5 * 10**3), 4, np.log10(2 * 10**4)])
ax.set_yticklabels([10**3, 2 * 10**3, 5 * 10**3, 10**4, 2 * 10**4])
ax.set_xlabel('Age', fontsize=14)
fig.text(0.78, -0.1, '(c) Recovered Features', fontsize=14, horizontalalignment='center')

fig.savefig('imgs/adult_2dim.png', bbox_inches='tight')
fig.savefig('imgs/adult_2dim.pdf', bbox_inches='tight')
