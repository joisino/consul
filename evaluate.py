import os
import numpy as np
import pickle
import argparse

from implicit.bpr import BayesianPersonalizedRanking
from implicit.nearest_neighbours import CosineRecommender
from scipy.sparse import csr_matrix


from methods import consul, oracle, PrivateRank, PrivateWalk

np.random.seed(0)


def recall(li, gt):
    if gt in li:
        return 1
    return 0


def nDCG(li, gt):
    if gt in li:
        return 1 / np.log2(li.tolist().index(gt) + 2)
    return 0


def list_minimum_group(li, sensitive):
    return np.bincount(sensitive[li], minlength=sensitive.max() + 1).min()


parser = argparse.ArgumentParser()
parser.add_argument('--data', choices=['100k', '1m', 'home', 'hetrec'], default='100k')
parser.add_argument('--prov', choices=['cosine', 'bpr'], default='cosine')
parser.add_argument('--sensitive', choices=['popularity', 'old'], default='popularity', help='`old` is valid only for MovieLens')
parser.add_argument('--split', type=int, default=1, help='Total number of parallel execusion (only for parallel execusion, set 1 otherwise)')
parser.add_argument('--block', type=int, default=0, help='Id of the current execusion (only for parallel excecusion, set 0 otherwise)')
args = parser.parse_args()
assert(args.sensitive == 'popularity' or args.data in ['100k', '1m'])
assert(0 <= args.block and args.block < args.split)

#
# Load Data
#

if args.data == '100k':
    n = 943
    m = 1682
    filename = 'ml-100k/u.data'
    delimiter = '\t'
elif args.data == '1m':
    n = 6040
    m = 3952
    filename = 'ml-1m/ratings.dat'
    delimiter = '::'

K = 10

if args.data == '100k' or args.data == '1m':
    raw_R = np.zeros((n, m))
    history = [[] for i in range(n)]
    with open(filename) as f:
        for r in f:
            user, movie, r, t = map(int, r.split(delimiter))
            user -= 1
            movie -= 1
            raw_R[user, movie] = r
            history[user].append((t, movie))
elif args.data == 'hetrec':
    raw_R = np.log2(np.load('hetrec.npy') + 1)
    n, m = raw_R.shape
    history = [[] for i in range(n)]
    for i in range(n):
        for j in np.nonzero(raw_R[i] > 0)[0]:
            history[i].append((np.random.rand(), j))
elif args.data == 'home':
    raw_R = np.load('Home_and_Kitchen.npy')
    n, m = raw_R.shape
    with open('Home_and_Kitchen_history.pickle', 'br') as f:
        history = pickle.load(f)

if args.sensitive == 'popularity':
    mask = raw_R > 0
    if args.data == '100k':
        sensitive = mask.sum(0) < 50
    elif args.data == '1m':
        sensitive = mask.sum(0) < 300
    elif args.data == 'hetrec':
        sensitive = mask.sum(0) < 50
    elif args.data == 'home':
        sensitive = mask.sum(0) < 50
    sensitive = sensitive.astype('int')
elif args.sensitive == 'old':
    sensitive = np.zeros(m, dtype='int')
    if args.data == '100k':
        filename = 'ml-100k/u.item'
        delimiter = '|'
    elif args.data == '1m':
        filename = 'ml-1m/movies.dat'
        delimiter = '::'
    with open(filename, encoding='utf8', errors='ignore') as f:
        for r in f:
            li = r.strip().split(delimiter)
            if '(19' in li[1]:
                year = 1900 + int(li[1].split('(19')[1].split(')')[0])
            elif '(20' in li[1]:
                year = 2000 + int(li[1].split('(20')[1].split(')')[0])
            sensitive[int(li[0]) - 1] = year < 1990


#
# Data Loaded
#

damping_factor = 0.01
tau = 5

provider_recall = 0
provider_nDCG = 0
provider_minimum = 0
oracle_recall = 0
oracle_nDCG = 0
oracle_minimum = 0
PR_recall = 0
PR_nDCG = 0
PR_minimum = 0
PW_recall = 0
PW_nDCG = 0
PW_minimum = 0
random_recall = 0
random_nDCG = 0
random_minimum = 0
consul_recall = 0
consul_nDCG = 0
consul_minimum = 0

PW_cnt = np.array(0)
consul_cnt = np.array(0)


start_index = int(n * args.block / args.split)
end_index = int(n * (args.block + 1) / args.split)

for i in range(start_index, end_index):
    gt = sorted(history[i])[-1][1]
    source = sorted(history[i])[-2][1]
    used = [y for x, y in history[i] if y != gt]

    R = raw_R.copy()
    R[i, gt] = 0

    mask = R > 0

    if args.prov == 'bpr':
        model = BayesianPersonalizedRanking(num_threads=1, random_state=0)
    elif args.prov == 'cosine':
        model = CosineRecommender()

    sR = csr_matrix(mask.T)
    model.fit(sR, show_progress=False)
    if args.prov == 'bpr':
        score = model.item_factors @ model.item_factors.T
    else:
        score = np.zeros((m, m))
        for item in range(m):
            for j, v in model.similar_items(item, m):
                score[item, j] = v

    score_remove = score.copy()
    score_remove[:, used] -= score.max() + 1
    score_remove -= np.eye(m) * (score.max() + 1)

    list_provider = np.argsort(-score_remove[source])[:K]
    provider_recall += recall(list_provider, gt)
    provider_nDCG += nDCG(list_provider, gt)
    provider_minimum += list_minimum_group(list_provider, sensitive)

    oracle_list = oracle(score_remove[source], sensitive, tau, used, K)
    oracle_recall += recall(oracle_list, gt)
    oracle_nDCG += nDCG(oracle_list, gt)
    oracle_minimum += list_minimum_group(oracle_list, sensitive)

    # Construct the recsys graph
    A = np.zeros((m, m))
    rank = np.argsort(-score_remove, 1)[:, :K]
    weight = 1 / np.log2(np.arange(K) + 2)
    weight /= weight.sum()
    A[np.arange(m).repeat(K), rank.reshape(-1)] += weight.repeat(m).reshape(K, m).T.reshape(-1)

    # Consul
    consul_list = consul(rank, sensitive, tau, source, used, K, access_conuter=consul_cnt)
    consul_recall += recall(consul_list, gt)
    consul_nDCG += nDCG(consul_list, gt)
    consul_minimum += list_minimum_group(consul_list, sensitive)

    # PrivateRank
    PR_list = PrivateRank(A, sensitive, tau, source, used, K, damping_factor)
    PR_recall += recall(PR_list, gt)
    PR_nDCG += nDCG(PR_list, gt)
    PR_minimum += list_minimum_group(PR_list, sensitive)

    # PrivateWalk
    PW_list = PrivateWalk(rank, sensitive, tau, source, used, K, access_conuter=PW_cnt)
    PW_recall += recall(PW_list, gt)
    PW_nDCG += nDCG(PW_list, gt)
    PW_minimum += list_minimum_group(PW_list, sensitive)

    # Random
    np.random.seed(0)
    random_score = np.random.rand(m)
    random_list = oracle(random_score, sensitive, tau, used, K)
    random_recall += recall(random_list, gt)
    random_nDCG += nDCG(random_list, gt)
    random_minimum += list_minimum_group(random_list, sensitive)

    print('#')
    print('# User {} - {}'.format(start_index, i))
    print('#')
    print('-' * 30)
    print('provider recall    {:.2f}'.format(provider_recall))
    print('oracle recall     ', oracle_recall)
    print('consul recall     ', consul_recall)
    print('PrivateRank recall', PR_recall)
    print('PrivateWalk recall', PW_recall)
    print('random recall     ', random_recall)
    print('-' * 30)
    print('provider nDCG    {:.2f}'.format(provider_nDCG))
    print('oracle nDCG     ', oracle_nDCG)
    print('consul nDCG     ', consul_nDCG)
    print('PrivateRank nDCG', PR_nDCG)
    print('PrivateWalk nDCG', PW_nDCG)
    print('random nDCG     ', random_nDCG)
    print('-' * 30)
    print('provider least count    {:.2f}'.format(provider_minimum))
    print('oracle least count     ', oracle_minimum)
    print('consul least count     ', consul_minimum)
    print('PrivateRank least count', PR_minimum)
    print('PrivateWalk least count', PW_minimum)
    print('random least count     ', random_minimum)
    print('-' * 30)
    print('consul access     ', consul_cnt)
    print('PrivateWalk access', PW_cnt)
    print('-' * 30)

if not os.path.exists('out'):
    os.mkdir('out')

with open('out/{}-{}-{}-{}.txt'.format(args.data, args.prov, args.sensitive, args.block), 'w') as f:
    print(provider_recall, file=f)
    print(provider_nDCG, file=f)
    print(provider_minimum, file=f)

    print(oracle_recall, file=f)
    print(oracle_nDCG, file=f)
    print(oracle_minimum, file=f)

    print(consul_recall, file=f)
    print(consul_nDCG, file=f)
    print(consul_minimum, file=f)
    print(consul_cnt, file=f)

    print(PR_recall, file=f)
    print(PR_nDCG, file=f)
    print(PR_minimum, file=f)

    print(PW_recall, file=f)
    print(PW_nDCG, file=f)
    print(PW_minimum, file=f)
    print(PW_cnt, file=f)

    print(random_recall, file=f)
    print(random_nDCG, file=f)
    print(random_minimum, file=f)
