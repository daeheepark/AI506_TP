import os
import sys
from itertools import combinations
from distutils.util import strtobool

import networkx as nx
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors, Word2Vec
from node2vec.node2vec.node2vec import Node2Vec
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

import numpy as np
from hypergraph import HyperGraph
from node2vec.node2vec.hypernode2vec import Hypernode2Vec

LOAD_PRETRAINED = False

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

if __name__ == "__main__":
    file_path = './project_data/paper_author.txt'
    lines = open(file_path, 'r').readlines()
    lineiter = iter(lines)

    line1 = next(lineiter)
    num_authors, num_pubs = list(map(int, line1.strip().split()))

    HG = HyperGraph()

    # for _ in range(num_pubs):
    #     linei = next(lineiter)
    #     authors = list(map(int, linei.strip().split()))
    #
    #     edge_list = [e for e in combinations(authors, 2)]
    #     HG.add_edges_from(edge_list)
    #     HG.update_hyperedges(authors)

    # # Load pretrained vectors, otherwise create a new graph
    # if os.path.exists("node2vec.kv") and LOAD_PRETRAINED:
    #     node_vectors = KeyedVectors.load("node2vec.kv", mmap='r')
    # else:
    #     G = nx.Graph()
    #     nodes, edges = set(), set()
    #     # one pass through paper_author.txt to get all authors->nodes and collaborations->edges
    #     for _ in range(num_pubs):
    #         linei = next(lineiter)
    #         authors = list(map(int, linei.strip().split()))
    #         nodes.update(authors)
    #         for combo in combinations(authors, 2):  # combinations find all pairs of author collaborations
    #             edges.add(tuple(sorted(combo)))     # sorting makes sure undirected edges are not double counted
    #
    #     G.add_nodes_from(nodes)
    #     G.add_edges_from(edges)
    #
    #     node2vec = Node2Vec(graph=G,
    #                         dimensions=64,
    #                         walk_length=10,
    #                         num_walks=100,
    #                         p=1,
    #                         q=1,
    #                         workers=8)
    #     model = node2vec.fit(window=10, min_count=1)
    #     node_vectors = model.wv
    #     model.save("node2vec.model")    # save model in case of more training later
    #     model.wv.save("node2vec.kv")    # keyed vectors for later use save memory by not loading entire model
    #     del model                       # save memory during computation

    # ## Hypernode2Vec
    # hypernode2vec = Hypernode2Vec(graph=HG,
    #                     dimensions=64,
    #                     walk_length=10,
    #                     num_walks=100,
    #                     p1=1,
    #                     p2=1,
    #                     workers=8)
    # model = hypernode2vec.fit(window=10, min_count=1)
    # node_vectors = model.wv
    # model.save("hypernode2vec.model")    # save model in case of more training later
    # model.wv.save("hypernode2vec.kv")    # keyed vectors for later use save memory by not loading entire model
    # del model                       # save memory during computation

    print("Load saved keyed vectors")
    node_vectors = KeyedVectors.load("hypernode2vec.kv")

    # # Visualize
    # print("Visualize")
    # tsne = TSNE(n_components=2).fit_transform(node_vectors.vectors)
    # plt.figure()
    # plt.scatter(tsne[:,0], tsne[:,1])
    # plt.savefig("hypernode2vec.png")
    # plt.close()

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
            if author1 in node_vectors.vocab.keys() and author2 in node_vectors.vocab.keys():   # check for fake authors
                sim += node_vectors.similarity(author1, author2)     # compute cosine similarity between two nodes
            nc += 1
        sim = sim/nc
        ans_lines.append(ans)
        sim_lines.append(sim)

    acc_best, thres_best, s, e = find_best_thres(ans_lines, sim_lines, 0., 1., 0.005)
    print(acc_best, thres_best)

    #     pred = 1 if sim >= sim_threshold else 0
    #
    #     if pred == ans:
    #         num_correct += 1
    #
    # print(f'acc: {num_correct/num_queries*100:.2f}')



    sys.exit()