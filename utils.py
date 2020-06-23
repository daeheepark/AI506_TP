from distutils.util import strtobool

import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
# from sklearn.manifold import TSNE
# from sklearn.cluster import KMeans


def load_data(query_file, label_file, node_vectors):
    qf = open(query_file, 'r')
    lf = open(label_file, 'r')

    num_queries = qf.readline()     # first line contains metadata
    queries, labels = [], []
    Q = []                          # query matrix containing query vectors
    input_dim = node_vectors.vectors.shape[1]
    
    for query in qf:
        queries.append(query.strip().split())
    
    for label in lf:
        labels.append(strtobool(label.strip()))

    for query in queries:
        query_vec = []
        
        for author_id in query:
            if author_id not in node_vectors:       # skip if it is fake
                continue    # TODO: or maybe give a zero vector
            node_vec = node_vectors[author_id]      # embedding lookup of author
            node_vec = torch.as_tensor(node_vec)    # numpy array -> tensor
            node_vec = node_vec.unsqueeze(0)        # shape (1 x input_dim)
            query_vec.append(node_vec)
        
        if len(query_vec) == 0:
            query_vec.append(torch.zeros([1, input_dim]))
        
        query_vec = torch.stack(query_vec, dim=0)
        q = torch.mean(query_vec, dim=0)        # method#1: column-wise average (best performance)
        # q = torch.prod(query_vec, dim=0)      # method#2: hadamard
        # q = torch.norm(query_vec, dim=0)      # method#3: frobenius norm
        Q.append(q)

    Q = torch.stack(Q, dim=0)
    Q = Q.squeeze(1)            # shape (query_size x input_dim)
    y = torch.tensor(labels)    # convert to tensor for torch computation

    # shuffles data and splits data into 80% train and 20% validation
    query_train, query_val, label_train, label_val = train_test_split(Q, y, test_size=0.2, random_state=0)

    qf.close()
    lf.close()

    return query_train, query_val, label_train, label_val


def plot_values(train_values, val_values, title, path):
    x = list(range(1, len(train_values)+1))
    plt.figure()
    plt.title(title)
    plt.plot(x, train_values, marker='o', label='Training')
    plt.plot(x, val_values, marker='x', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.tight_layout()
    plt.legend()
    plt.savefig(path)


def analyze():
    # tsne = TSNE(n_components=2).fit_transform(node_vectors.vectors)
    # plt.figure()
    # plt.scatter(tsne[:,0], tsne[:,1])
    # plt.savefig("node2vec.png")
    # plt.close()
    pass