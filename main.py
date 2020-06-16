import os
import sys
import time
from itertools import combinations, permutations
from distutils.util import strtobool

import networkx as nx
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors, Word2Vec
from node2vec.node2vec.node2vec import Node2Vec
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

import numpy as np
import pandas as pd
from hypergraph import HyperGraph
from node2vec.node2vec.hypernode2vec import Hypernode2Vec

LOAD_PRETRAINED = False
RESULT_DIR = './result'

def find_best_thres(ans_list, pred_list, start, end, intv):
    acc_best = 0
    for thres in np.arange(start, end, intv):
        pred_list_th = np.array(pred_list) > thres
        correct = 0
        for gt, pred in zip(ans_list, list(pred_list_th)):
            if gt == pred:
                correct += 1
        acc = correct / len(ans_list)
        if acc_best < acc:
            acc_best = acc
            thres_best = thres
            intv_s, intv_e = thres, thres + intv

    return acc_best, thres_best, intv_s, intv_e

def evaluate(node_vectors):
    print("Evaluating")
    qp_lines = open('./project_data/query_public.txt', 'r').readlines()
    ap_lines = open('./project_data/answer_public.txt', 'r').readlines()
    qp_iter = iter(qp_lines)
    ap_iter = iter(ap_lines)

    num_queries = int(next(qp_iter))
    num_correct = 0

    sim_lines = []
    ans_lines = []

    for _ in range(num_queries):
        query = next(qp_iter)
        query = query.strip().split()
        ans = next(ap_iter).strip()
        ans = strtobool(ans)
        sim = 0.
        nc = 0
        # sim_threshold = 0.5

        for author1, author2 in combinations(query, 2):
            if author1 in node_vectors.vocab.keys() and author2 in node_vectors.vocab.keys():  # check for fake authors
                sim += node_vectors.similarity(author1, author2)  # compute cosine similarity between two nodes
            nc += 1
        sim = sim / nc
        ans_lines.append(ans)
        sim_lines.append(sim)

    acc_best, thres_best, s, e = find_best_thres(ans_lines, sim_lines, 0., 1., 0.005)
    print(acc_best, thres_best)

    return acc_best, thres_best

def visualize(node_vectors, p1, p2, p, q):
    # Visualize
    print("Visualize")
    tsne = TSNE(n_components=2).fit_transform(node_vectors.vectors)
    plt.figure()
    plt.scatter(tsne[:,0], tsne[:,1])
    plt.savefig(os.path.join(RESULT_DIR, f"hypernode2vec_p({p})q({q})_p1({p1})p2({p2}).png"))
    plt.close()

def genHypernode2vec(HG, p1, p2, p, q, is_load=None):

    if is_load is None:
        hypernode2vec = Hypernode2Vec(graph=HG,
                                      dimensions=64,
                                      walk_length=10,
                                      num_walks=100,
                                      p1=p1,
                                      p2=p2,
                                      p=p, q=q,
                                      workers=8)
        model = hypernode2vec.fit(window=10, min_count=1)
        node_vectors = model.wv
        model.save(os.path.join(RESULT_DIR, f"hypernode2vec_p({p})q({q})_p1({p1})p2({p2}).model"))  # save model in case of more training later
        model.wv.save(os.path.join(RESULT_DIR, f"hypernode2vec_p({p})q({q})_p1({p1})p2({p2}).kv"))  # keyed vectors for later use save memory by not loading entire model
        del model  # save memory during computation

    elif is_load is not None and os.path.isfile(is_load):
        print("Load saved keyed vectors")
        node_vectors = KeyedVectors.load("hypernode2vec.kv")

    elif is_load is not None and not os.path.isfile(is_load):
        raise KeyError('node vector file does not exist')

    return node_vectors

if __name__ == "__main__":
    if not os.path.isdir(RESULT_DIR):
        os.mkdir(RESULT_DIR)
    file_path = './project_data/paper_author.txt'
    lines = open(file_path, 'r').readlines()
    lineiter = iter(lines)

    line1 = next(lineiter)
    num_authors, num_pubs = list(map(int, line1.strip().split()))

    HG = HyperGraph()

    for _ in range(num_pubs):
        linei = next(lineiter)
        authors = list(map(int, linei.strip().split()))

        edge_list = [e for e in combinations(authors, 2)]
        HG.add_edges_from(edge_list)
        HG.update_hyperedges(authors)

    pq = [(1,1), (1,0.5), (1,2)] # [DeepWalk, reflecting homophily, reflecting structural equivalence]
    p1p2 = [(1,1), (1,0.5), (1,2)]

    result = []
    num_run = int(len(pq) * len(p1p2))
    current = 0
    for p, q in pq:
        for p1, p2 in p1p2:
            current += 1
            print(f"p: {p}, q: {q}, p1: {p1}, p2: {p2} start ({current}/{num_run})")

            node_vectors = genHypernode2vec(HG, p1=p1, p2=p2, p=p, q=q)

            # visualize(node_vectors, p1=p1, p2=p2, p=p, q=q)
            acc_best, thres_best = evaluate(node_vectors)
            result.append({'p':p, 'q':q, 'p1':p1, 'p2':p2, 'thres_best':thres_best, 'acc_best':acc_best})

    print('Saving final result')
    pddf = pd.DataFrame(result)
    pddf.to_csv(os.path.join(RESULT_DIR, "result_%s.csv" % time.strftime("%Y%m%d_%H:%M:%S")))

    sys.exit()