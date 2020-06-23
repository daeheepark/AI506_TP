import os
from itertools import combinations

import networkx as nx
from node2vec import Node2Vec


def embed_nodes(graph_file, kv_path, p=1, q=1):
    f = open(graph_file, 'r')
    lines = f.readlines()
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
    
    f.close()

    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    node2vec = Node2Vec(graph=G,
                        dimensions=64,
                        walk_length=10,
                        num_walks=100,
                        p=p,
                        q=q,
                        workers=2)
    model = node2vec.fit(window=10, min_count=1)
    # node_vectors = model.wv
    # model.save("./kvs/node2vec_p("+str(p)+")q("+str(q)+").model")    # save model in case of more training later
    model.wv.save(kv_path)    # keyed vectors for later use save memory by not loading entire model
    del model   # save memory during computation
    del G                           


if __name__ == "__main__":
    graph_data = './project_data/paper_author.txt'
    p, q = 1, 0.5     # set hyperparameters p and q
    kv_path = "./kvs/node2vec_p("+str(p)+")q("+str(q)+").kv"
    if os.path.exists(kv_path):
        print("Keyed vectors already exist at", kv_path)
    else:
        print("Creating node2vec graph...")
        embed_nodes(graph_data, kv_path, p=p, q=q)
