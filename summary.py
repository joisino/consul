import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', choices=['100k', '1m', 'home', 'hetrec'], default='100k')
parser.add_argument('--prov', choices=['cosine', 'bpr'], default='cosine')
parser.add_argument('--sensitive', choices=['popularity', 'old'], default='popularity', help='`old` is valid only for MovieLens')
parser.add_argument('--split', type=int, default=1, help='Total number of parallel execusion (only for parallel execusion, set 1 otherwise)')
args = parser.parse_args()

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

PW_cnt = 0
consul_cnt = 0


for i in range(args.split):
    with open('out/{}-{}-{}-{}.txt'.format(args.data, args.prov, args.sensitive, i)) as f:
        provider_recall += float(f.readline())
        provider_nDCG += float(f.readline())
        provider_minimum += float(f.readline())

        oracle_recall += float(f.readline())
        oracle_nDCG += float(f.readline())
        oracle_minimum += float(f.readline())

        consul_recall += float(f.readline())
        consul_nDCG += float(f.readline())
        consul_minimum += float(f.readline())
        consul_cnt += float(f.readline())

        PR_recall += float(f.readline())
        PR_nDCG += float(f.readline())
        PR_minimum += float(f.readline())

        PW_recall += float(f.readline())
        PW_nDCG += float(f.readline())
        PW_minimum += float(f.readline())
        PW_cnt += float(f.readline())

        random_recall += float(f.readline())
        random_nDCG += float(f.readline())
        random_minimum += float(f.readline())


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
