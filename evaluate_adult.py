import numpy as np
import argparse

from scipy.sparse import load_npz

from methods import consul, oracle, PrivateRank, PrivateWalk


def list_minimum_group(li, sensitive):
    return np.bincount(sensitive[li], minlength=sensitive.max() + 1).min()


np.random.seed(0)


parser = argparse.ArgumentParser()
parser.add_argument('--split', type=int, default=1, help='Total number of parallel execusion (only for parallel execusion, set 1 otherwise)')
parser.add_argument('--block', type=int, default=0, help='Id of the current execusion (only for parallel excecusion, set 0 otherwise)')
args = parser.parse_args()
assert(0 <= args.block and args.block < args.split)

#
# Load Data
#

X = np.load('adult_X.npy')
y = np.load('adult_y.npy')
sensitive = np.load('adult_a.npy')

m, d = X.shape
K = 10

weight = 1 / np.log2(np.arange(K)[::-1] + 2)
weight /= weight.sum()

R = np.load('adult_R.npy')
At = load_npz('adult_At.npz')
rank = np.load('adult_rank.npy')

#
# Data Loaded
#

damping_factor = 0.01
tau = 5

provider_accuracy = 0
provider_minimum = 0
oracle_accuracy = 0
oracle_minimum = 0
PR_accuracy = 0
PR_minimum = 0
PW_accuracy = 0
PW_minimum = 0
random_accuracy = 0
random_minimum = 0
consul_accuracy = 0
consul_minimum = 0

PW_cnt = np.array(0)
consul_cnt = np.array(0)

start_index = int(m * args.block / args.split)
end_index = int(m * (args.block + 1) / args.split)

for i in range(start_index, end_index):
    source = i
    used = [source]

    # Service provider's recsys
    list_provider = rank[source]
    provider_accuracy += (y[list_provider] == y[source]).sum()
    provider_minimum += list_minimum_group(list_provider, sensitive)

    oracle_list = oracle(R[source], sensitive, tau, used, K)
    oracle_accuracy += (y[oracle_list] == y[source]).sum()
    oracle_minimum += list_minimum_group(oracle_list, sensitive)

    # Consul
    consul_list = consul(rank, sensitive, tau, source, used, K, access_conuter=consul_cnt)
    consul_accuracy += (y[consul_list] == y[source]).sum()
    consul_minimum += list_minimum_group(consul_list, sensitive)

    # PrivateRank
    PR_list = PrivateRank(At.T, sensitive, tau, source, used, K, damping_factor)
    PR_accuracy += (y[PR_list] == y[source]).sum()
    PR_minimum += list_minimum_group(PR_list, sensitive)

    # PrivateWalk
    PW_list = PrivateWalk(rank, sensitive, tau, source, used, K, access_conuter=PW_cnt)
    PW_accuracy += (y[PW_list] == y[source]).sum()
    PW_minimum += list_minimum_group(PW_list, sensitive)

    # Random
    np.random.seed(0)
    random_score = np.random.rand(m)
    random_list = oracle(random_score, sensitive, tau, used, K)
    random_accuracy += (y[random_list] == y[source]).sum()
    random_minimum += list_minimum_group(random_list, sensitive)

    print('#')
    print('# User {} - {}'.format(start_index, i))
    print('#')
    print('-' * 30)
    print('provider accuracy    {:.2f}'.format(provider_accuracy))
    print('oracle accuracy     ', oracle_accuracy)
    print('consul accuracy     ', consul_accuracy)
    print('PrivateRank accuracy', PR_accuracy)
    print('PrivateWalk accuracy', PW_accuracy)
    print('random accuracy     ', random_accuracy)
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


with open('out/adult-{}.txt'.format(args.block), 'w') as f:
    print(provider_accuracy, file=f)
    print(provider_minimum, file=f)

    print(oracle_accuracy, file=f)
    print(oracle_minimum, file=f)

    print(consul_accuracy, file=f)
    print(consul_minimum, file=f)
    print(consul_cnt, file=f)

    print(PR_accuracy, file=f)
    print(PR_minimum, file=f)

    print(PW_accuracy, file=f)
    print(PW_minimum, file=f)
    print(PW_cnt, file=f)

    print(random_accuracy, file=f)
    print(random_minimum, file=f)
