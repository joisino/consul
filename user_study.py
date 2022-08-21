import re
import numpy as np

from implicit.bpr import BayesianPersonalizedRanking
from scipy.sparse import csr_matrix
from collections import defaultdict

from methods import consul


np.random.seed(0)


n = 6040
m = 3952
K_prov = 10
K_show = 6
filename = 'ml-1m/ratings.dat'
delimiter = '::'

R = np.zeros((n, m))
history = [[] for i in range(n)]
with open('ml-1m/ratings.dat', encoding='utf8', errors='ignore') as f:
    for r in f:
        user, movie, r, t = map(int, r.split(delimiter))
        user -= 1
        movie -= 1
        R[user, movie] = r
        history[user].append((t, movie))

movie_to_year = defaultdict(int)
movie_to_tag = {}
movie_title = {}
tag_to_movie = defaultdict(list)
with open('ml-1m/movies.dat', encoding='utf8', errors='ignore') as f:
    for r in f:
        movie, title, tags = r.split(delimiter)
        movie = int(movie) - 1
        tags = tags.strip().split('|')
        movie_to_tag[movie] = tags
        if ', The' in title:
            title = 'The ' + title.replace(', The', '')
        if ', A' in title:
            title = 'A ' + title.replace(', A', '')
        movie_title[movie] = title
        for tag in tags:
            tag_to_movie[tag].append(movie)
        match = re.match(r'.*\(([0-9]*)\)', title)
        movie_to_year[movie] = int(match[1])

mask = R > 0

model = BayesianPersonalizedRanking(num_threads=1, random_state=0)
sR = csr_matrix(mask.T)
model.fit(sR, show_progress=False)
score = model.item_factors @ model.item_factors.T
score -= (score.max() + 1) * np.eye(m)

rank = np.argsort(-score, 1)[:, :K_prov]
rank = np.argsort(-score, 1)[:, :K_prov]

np.save('ml1m_score.npy', score)
np.save('ml1m_rank.npy', rank)

score = np.load('ml1m_score.npy')
rank = np.load('ml1m_rank.npy')

count = mask.sum(0)

cnt = 0
with open('AMT_CONSUL.csv', 'w') as f:
    print(','.join(['"source"'] + ['"rec1{}"'.format(i + 1) for i in range(K_show)] + ['"rec2{}"'.format(i + 1) for i in range(K_show)] + ['"order"']), file=f)
    pop_rank = np.argsort(-count)
    for ind, source in enumerate(pop_rank[:100]):
        print('source', source, movie_title[source])

        list_provider = np.argsort(-score[source])[:K_show]
        for x in list_provider:
            print('provider', x, movie_title[x])

        sensitive = [
            0 if (movie_to_year[source] - 10 <= movie_to_year[i] and movie_to_year[i] <= movie_to_year[source] + 10) else 1 for i in range(m)
        ]

        tau = 3
        used = [source]
        list_consul = consul(rank, sensitive, tau, source, used, K_show, max_length=100, seed=0)

        for x in list_consul:
            print('CONSUL', x, movie_title[x])

        if (list_provider == list_consul).all():
            cnt += 1
        print(cnt)

        order = ind % 2
        print('"{}"'.format(movie_title[source]), end='', file=f)
        if order == 0:
            list = np.concatenate([list_consul, list_provider])
        else:
            list = np.concatenate([list_provider, list_consul])
        for x in list:
            print(',"{}"'.format(movie_title[x]), end='', file=f)
        print(',"{}"'.format(order), file=f)
