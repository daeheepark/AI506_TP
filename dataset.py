import torch


class QueryDataset(torch.utils.data.Dataset):
    def __init__(self, queries, labels):
        self.queries = queries
        self.labels = labels

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return (self.queries[idx], self.labels[idx])
