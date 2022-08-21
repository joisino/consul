import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=int, default=1, help='Total number of parallel execusion (only for parallel execusion, set 1 otherwise)')
args = parser.parse_args()

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

PW_cnt = 0
consul_cnt = 0

for i in range(args.split):
    with open('out/adult-{}.txt'.format(i)) as f:
        provider_accuracy += float(f.readline())
        provider_minimum += float(f.readline())

        oracle_accuracy += float(f.readline())
        oracle_minimum += float(f.readline())

        consul_accuracy += float(f.readline())
        consul_minimum += float(f.readline())
        consul_cnt += float(f.readline())

        PR_accuracy += float(f.readline())
        PR_minimum += float(f.readline())

        PW_accuracy += float(f.readline())
        PW_minimum += float(f.readline())
        PW_cnt += float(f.readline())

        random_accuracy += float(f.readline())
        random_minimum += float(f.readline())


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
