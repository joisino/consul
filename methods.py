import subprocess

import numpy as np


def oracle(score, sensitive, tau, used, K):
    n_sensitive = len(set(sensitive))
    rec_list = []
    cnt = [0 for _ in range(n_sensitive)]
    sum_res = tau * n_sensitive
    score_remove = score.copy()
    score_remove[used] -= score.max() + 1
    for j in np.argsort(-score_remove):
        if j not in used + rec_list and sum_res - max(0, tau - cnt[sensitive[j]]) <= K - len(rec_list) - 1:
            rec_list.append(j)
            if cnt[sensitive[j]] < tau:
                sum_res -= 1
            cnt[sensitive[j]] += 1
        if len(rec_list) == K:
            return np.array(rec_list)


def PrivateRank(A, sensitive, tau, source, used, K, damping_factor, n_iter=10):
    psc = np.zeros(A.shape[0])
    cur = np.zeros(A.shape[0])
    cur[source] = 1
    for _ in range(n_iter + 1):
        psc += (1 - damping_factor) * cur
        cur = damping_factor * A.T @ cur
    return oracle(psc, sensitive, tau, used, K)


def PrivateWalk(rank, sensitive, tau, source, used, K, access_conuter=0, max_length=100, seed=0):
    np.random.seed(seed)
    n_sensitive = len(set(sensitive))
    rec_list = []
    cnt = [0 for _ in range(n_sensitive)]
    sum_res = tau * n_sensitive
    weight = 1 / np.log2(np.arange(K) + 2)
    weight /= weight.sum()
    for _ in range(K):
        cur = source
        for _ in range(max_length):
            access_conuter += 1
            cur = np.random.choice(rank[cur], p=weight)
            if cur not in used + rec_list and sum_res - max(0, tau - cnt[sensitive[cur]]) <= K - len(rec_list) - 1:
                break
        while cur in used + rec_list or sum_res - max(0, tau - cnt[sensitive[cur]]) > K - len(rec_list) - 1:
            cur = np.random.randint(rank.shape[0])
        rec_list.append(cur)
        if cnt[sensitive[cur]] < tau:
            sum_res -= 1
        cnt[sensitive[cur]] += 1
    return np.array(rec_list)


def consul(rank, sensitive, tau, source, used, K, access_conuter=0, max_length=100, seed=0):
    np.random.seed(seed)
    n_sensitive = len(set(sensitive))
    rec_list = []
    cnt = [0 for _ in range(n_sensitive)]
    sum_res = tau * n_sensitive
    queue = [source]
    visited = [False for i in range(rank.shape[0])]
    for _ in range(max_length):
        while True:
            if len(queue) == 0:
                cur = None
                break
            cur = queue.pop(-1)
            if not visited[cur]:
                visited[cur] = True
                break
        if cur is None:
            break
        access_conuter += 1
        for j in rank[cur]:
            if j not in used + rec_list and sum_res - max(0, tau - cnt[sensitive[j]]) <= K - len(rec_list) - 1:
                rec_list.append(j)
                if cnt[sensitive[j]] < tau:
                    sum_res -= 1
                cnt[sensitive[j]] += 1
            if len(rec_list) == K:
                break
        if len(rec_list) == K:
            break
        for i in rank[cur][::-1]:
            if not visited[i]:
                queue.append(i)
    while len(rec_list) < K:
        j = np.random.randint(rank.shape[0])
        if j not in used + rec_list and sum_res - max(0, tau - cnt[sensitive[j]]) <= K - len(rec_list) - 1:
            rec_list.append(j)
            if cnt[sensitive[j]] < tau:
                sum_res -= 1
            cnt[sensitive[j]] += 1
    return np.array(rec_list)


def LOE(rank, sensitive, tau, source, used, K, factors=100):
    n = rank.shape[0]
    with open('A.txt', 'w') as f:
        for i in range(n):
            ss = set(rank[i].tolist())
            for j in range(n):
                if j in ss:
                    print(1, end=' ', file=f)
                else:
                    print(0, end=' ', file=f)
            print(file=f)

    R_script = """install.packages('loe')
library(MASS)
library(loe)
A = scan('A.txt')
A = t(matrix(A, sqrt(length(A)), sqrt(length(A))))
res <- LOE(ADM=A, p={})
write(res$X, 'res.txt')
""".format(factors)

    with open('LOE.R', 'w') as f:
        print(R_script, file=f)

    subprocess.call('docker run -ti --rm -v "$PWD":/home/docker -w /home/docker --net host r-base Rscript LOE.R', shell=True)

    with open('res.txt') as f:
        x = np.array(list(map(float, f.read().split()))).reshape(factors, n).T

    distance = ((x - x[source]) ** 2).sum(1)
    n_sensitive = len(set(sensitive))
    rec_list = []
    cnt = [0 for _ in range(n_sensitive)]
    sum_res = tau * n_sensitive
    for j in np.argsort(distance):
        if j not in used + rec_list and sum_res - max(0, tau - cnt[sensitive[j]]) <= K - len(rec_list) - 1:
            rec_list.append(j)
            if cnt[sensitive[j]] < tau:
                sum_res -= 1
            cnt[sensitive[j]] += 1
        if len(rec_list) == K:
            break
    return np.array(rec_list)
