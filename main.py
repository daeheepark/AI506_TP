import os
import sys
from itertools import combinations

import networkx as nx
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from node2vec import Node2Vec
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

from hypergraph import HyperGraph


if __name__ == "__main__":
    filePath = './project_data/paper_author.txt'
    lines = open(filePath, 'r').readlines()
    lineiter = iter(lines)

    line1 = next(lineiter)
    num_authors, num_pubs = list(map(int, line1.strip().split()))

    HG = HyperGraph()

    # for _ in range(num_pubs):
    #     linei = next(lineiter)
    #     authors = list(map(int, linei.strip().split()))
    #     HG.update_line(authors)

    # Load pretrained vectors, otherwise create a new graph
    if os.path.exists("node2vec.kv"):
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

    tsne = TSNE(n_components=2).fit_transform(node_vectors.vectors)
    plt.figure()
    plt.scatter(tsne[:,0], tsne[:,1])
    plt.savefig("node2vec.png")
    plt.close()

    sys.exit()