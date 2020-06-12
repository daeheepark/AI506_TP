import os
import sys
import numpy as np
# import torch
from itertools import combinations

from gensim.models import KeyedVectors
# from node2vec.node2vec.node2vec import Node2Vec

from hypergraph import HyperGraph
from _node2vec import load_data, embed_nodes
from classifier import Classifier

LOAD_PRETRAINED = True

if __name__ == "__main__":
    graph_data = './project_data/paper_author.txt'
    query_data = './project_data/query_public.txt'
    label_data = './project_data/answer_public.txt'
    test_data = './project_data/query_private.txt'

    # Split and Shuffle Data
    query_train, query_val, label_train, label_val = load_data(query_data, label_data)

    # HG = HyperGraph()
    # for _ in range(num_pubs):
    #     linei = next(lineiter)
    #     authors = list(map(int, linei.strip().split()))
    #     HG.update_line(authors)

    # Load pretrained vectors, otherwise create a new graph
    if os.path.exists("node2vec.kv") and LOAD_PRETRAINED:
        node_vectors = KeyedVectors.load("node2vec.kv", mmap='r')
    else:
        node_vectors = embed_nodes(graph_data)

    num_correct = 0
    for i in range(len(query_train)):
        sim = 0.
        sim_threshold = 0.5
        sim_count = 0

        for author1, author2 in combinations(query_train[i], 2):
            if author1 in node_vectors.vocab.keys() and author2 in node_vectors.vocab.keys():   # check for fake authors
                sim += node_vectors.similarity(author1, author2)     # compute cosine similarity between two nodes
                sim_count += 1

        if sim_count > 0:
            sim /= sim_count

        pred = 1 if sim >= sim_threshold else 0

        if pred == label_train[i]:
            num_correct += 1

    print(f'acc: {num_correct/len(query_train)*100:.2f}')

    # Train 
    # classifier = Classifier()
    # loss_history = classifier.train(node_vectors, query_train, label_train)
    # label_train_pred = classifier.predict(query_train)
    # label_val_pred = classifier.predict(query_val)
    # train_acc = np.mean(label_train == label_train_pred)
    # val_acc = np.mean(label_val == label_val_pred)



    sys.exit()
