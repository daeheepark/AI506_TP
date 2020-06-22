import os
import sys
from itertools import combinations
from distutils.util import strtobool

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from gensim.models import KeyedVectors
from tqdm import tqdm, trange

from node2vec.node2vec.hypernode2vec import Hypernode2Vec
from hypergraph import HyperGraph
from dataset import QueryDataset
from model import SimpleNN

LOAD_PRETRAINED = False
RESULT_DIR = './result'

def visualize(node_vectors, p1, p2, p, q):
    # Visualize
    print("Visualize")
    tsne = TSNE(n_components=2).fit_transform(node_vectors.vectors)
    plt.figure()
    plt.scatter(tsne[:, 0], tsne[:, 1])
    plt.savefig(os.path.join(RESULT_DIR, f"hypernode2vec_p({p})q({q})_p1({p1})p2({p2}).png"))
    plt.close()


def genHypernode2vec(HG, p1, p2, p, q, num_dim, is_load=None):
    if is_load is None:
        hypernode2vec = Hypernode2Vec(graph=HG,
                                      dimensions=num_dim,
                                      walk_length=10,
                                      num_walks=100,
                                      p1=p1,
                                      p2=p2,
                                      p=p, q=q,
                                      workers=8)
        model = hypernode2vec.fit(window=10, min_count=1)
        node_vectors = model.wv
        # model.save(os.path.join(RESULT_DIR,
        #                         f"hypernode2vec_p({p})q({q})_p1({p1})p2({p2}).model"))  # save model in case of more training later
        model.wv.save(os.path.join(RESULT_DIR,
                                   f"hypernode2vec_p({p})q({q})_p1({p1})p2({p2}dim({num_dim})).kv"))  # keyed vectors for later use save memory by not loading entire model
        del model  # save memory during computation

    elif is_load is not None and os.path.isfile(is_load):
        print("Load saved keyed vectors")
        node_vectors = KeyedVectors.load(is_load)

    elif is_load is not None and not os.path.isfile(is_load):
        raise KeyError('node vector file does not exist')

    return node_vectors



if __name__ == "__main__":
    if not os.path.isdir(RESULT_DIR):
        os.mkdir(RESULT_DIR)

    graph_data = './project_data/paper_author.txt'
    query_data = './project_data/query_public.txt'
    label_data = './project_data/answer_public.txt'
    test_data = './project_data/query_private.txt'

    graph_iter = iter(open(graph_data, 'r').readlines())

    # hypernode2vec creation
    HG = HyperGraph()
    line1 = next(graph_iter)

    num_authors, num_pubs = list(map(int, line1.strip().split()))

    for _ in range(num_pubs):
        linei = next(graph_iter)
        authors = list(map(int, linei.strip().split()))

        edge_list = [e for e in combinations(authors, 2)]
        HG.add_edges_from(edge_list)
        HG.update_hyperedges(authors)

    pq = [(1,1), (1,0.5), (1,2)] # [DeepWalk, reflecting homophily, reflecting structural equivalence]
    p1p2 = [(1,1), (1,0.5), (1,2)]
    dims = [128, 256, 512]

    result = []
    num_run = int(len(pq) * len(p1p2) * len(dims))
    current = 0
    for p, q in pq:
        for p1, p2 in p1p2:
            for dim in dims:
                current += 1
                print(f"p: {p}, q: {q}, p1: {p1}, p2: {p2}, dim: {dim}, start ({current}/{num_run})")

                node_vectors = genHypernode2vec(HG, p1=p1, p2=p2, p=p, q=q, num_dim=dim)


