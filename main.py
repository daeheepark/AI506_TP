import os
import sys
from itertools import combinations
from distutils.util import strtobool

import networkx as nx
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from node2vec.node2vec.node2vec import Node2Vec
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

from hypergraph import HyperGraph

LOAD_PRETRAINED = False

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
    #     HG.update_line(authors)

    # Load pretrained vectors, otherwise create a new graph
    if os.path.exists("node2vec.kv") and LOAD_PRETRAINED:
        node_vectors = KeyedVectors.load("node2vec.kv", mmap='r')
    else:
        G = nx.Graph()
        nodes, edges = set(), set()
        # one pass through paper_author.txt to get all authors->nodes and collaborations->edges
        for _ in range(num_pubs):
            linei = next(lineiter)
            authors = list(map(int, linei.strip().split()))
            nodes.update(authors)
            for combo in combinations(authors, 2):  # combinations find all pairs of author collaborations
                edges.add(tuple(sorted(combo)))     # sorting makes sure undirected edges are not double counted
        
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)

        node2vec = Node2Vec(graph=G,
                            dimensions=64,
                            walk_length=10,
                            num_walks=100,
                            p=1,
                            q=1,
                            workers=2)
        model = node2vec.fit(window=10, min_count=1)
        node_vectors = model.wv
        model.save("node2vec.model")    # save model in case of more training later
        model.wv.save("node2vec.kv")    # keyed vectors for later use save memory by not loading entire model
        del model                       # save memory during computation

    # tsne = TSNE(n_components=2).fit_transform(node_vectors.vectors)
    # plt.figure()
    # plt.scatter(tsne[:,0], tsne[:,1])
    # plt.savefig("node2vec.png")
    # plt.close()

    qp_lines = open('./project_data/query_public.txt', 'r').readlines()
    ap_lines = open('./project_data/answer_public.txt', 'r').readlines()
    qp_iter = iter(qp_lines)
    ap_iter = iter(ap_lines)

    num_queries = int(next(qp_iter))
    num_correct = 0

    for _ in range(num_queries):
        query = next(qp_iter)
        query = query.strip().split()
        ans = next(ap_iter).strip()
        ans = strtobool(ans)
        sim = 0.
        sim_threshold = 0.5

        for author1, author2 in combinations(query, 2):
            if author1 in node_vectors.vocab.keys() and author2 in node_vectors.vocab.keys():   # check for fake authors
                sim += node_vectors.similarity(author1, author2)     # compute cosine similarity between two nodes

        pred = 1 if sim >= sim_threshold else 0

        if pred == ans:
            num_correct += 1

    print(f'acc: {num_correct/num_queries*100:.2f}')

    sys.exit()