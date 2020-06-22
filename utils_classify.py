import torch
from torch.utils.data import Dataset

from gensim.models import KeyedVectors
from distutils.util import strtobool

import numpy as np

class CoauthorshipDataset(Dataset):
    def __init__(self, query_fn, label_fn, kv_fn):

        self.queries, self.labels, self.num_queries = load_qnl(query_fn, label_fn)
        self.kv = KeyedVectors.load(kv_fn, mmap='r')
        self.meanvector = np.mean(self.kv.vectors, 0)
        # self.query_vecs = []

    def __len__(self):
        return self.num_queries

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        query_line = self.queries[idx]
        query_vec = torch.Tensor().reshape(-1,self.kv.vector_size)

        for author_id in query_line:
            if author_id not in self.kv:  # skip if it is fake
                continue
                # node_vec = self.meanvector  # TODO: or maybe give a zero vector
            else:
                node_vec = self.kv[author_id]  # embedding lookup of author
            node_vec = torch.as_tensor(node_vec)  # numpy array -> tensor
            node_vec = node_vec.unsqueeze(0)  # shape (1 x input_dim)
            query_vec = torch.cat((query_vec, node_vec), 0)

        return query_vec, torch.tensor(self.labels[idx], dtype=torch.float)

def load_qnl(query_file, label_file):
    qf = open(query_file, 'r')
    lf = open(label_file, 'r')

    num_queries = int(qf.readline())  # first line contains metadata
    queries, labels = [], []

    for query in qf:
        queries.append(query.strip().split())

    for label in lf:
        labels.append(strtobool(label.strip()))

    qf.close()
    lf.close()

    return queries, labels, num_queries