from itertools import combinations
from distutils.util import strtobool

import networkx as nx
from sklearn.model_selection import train_test_split
# from sklearn.manifold import TSNE
# from sklearn.cluster import KMeans
from node2vec import Node2Vec


def load_data(query_file, label_file):
    qf = open(query_file, 'r')
    lf = open(label_file, 'r')

    num_queries = qf.readline()     # first line contains metadata
    queries, labels = [], []
    
    for query in qf:
        queries.append(query.strip().split())
    
    for label in lf:
        labels.append(strtobool(label.strip()))

    # shuffles data and splits data into 80% train and 20% validation
    query_train, query_val, label_train, label_val = train_test_split(queries, labels, test_size=0.2, random_state=0)

    qf.close()
    lf.close()

    return query_train, query_val, label_train, label_val


def embed_nodes(graph_file):
    lines = open(graph_file, 'r').readlines()
    lineiter = iter(lines)
    line1 = next(lineiter)
    num_authors, num_pubs = list(map(int, line1.strip().split()))

    G = nx.Graph()
    nodes, edges = set(), set()

    # one pass through paper_author.txt to get all authors->nodes and collaborations->edges
    for _ in range(num_pubs):
        linei = next(lineiter)
        authors = list(map(int, linei.strip().split()))
        nodes.update(authors)
        for combo in combinations(authors, 2):  # combinations find all pairs of author collaborations
            edges.add(tuple(sorted(combo)))     # sorting makes sure undirected edges are not double counted
    
    graph_file.close()

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
    del G                           

    return node_vectors


def analyze():
    # tsne = TSNE(n_components=2).fit_transform(node_vectors.vectors)
    # plt.figure()
    # plt.scatter(tsne[:,0], tsne[:,1])
    # plt.savefig("node2vec.png")
    # plt.close()
    pass