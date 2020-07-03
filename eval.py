import torch
from torch.utils.data import DataLoader

from gensim.models import KeyedVectors

from utils import load_test_data
from dataset import QueryDataset
from model import SimpleNN

from tqdm import tqdm


def predict(model, test_set):
    f = open("answer_private.txt", "w")
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
    model.eval()
    with torch.no_grad():
        for queries, _ in tqdm(test_loader, desc="Testing Iteration"):
            scores = model(queries)
            _, preds = torch.max(scores.data, dim=1)
            preds = preds == 1
            for pred in preds:
                f.write(f"{pred.item()}\n")
    f.close()


if __name__ == "__main__":
    test_data = './project_data/query_private.txt'
    pretrained_kv_path = "./kvs/hypernode2vec_p(1)q(2)_p1(1)p2(0.5)dim(512).kv"
    pretrained_model_path = "./models/hypernode2vec_p(1)q(2)_p1(1)p2(0.5)dim(512)_hid(256)_hid2(64)_dropout_bn.pth"
    
    print("Loading pretrained keyed vectors...")
    node_vectors = KeyedVectors.load(pretrained_kv_path, mmap='r')
    query_test = load_test_data(test_data, node_vectors)
    test_set = QueryDataset(query_test, labels=[0]*len(query_test))
    
    print("Loading pretrained model...")
    model = SimpleNN(input_dim=512, hidden_dim=256)
    model.load_state_dict(torch.load(pretrained_model_path), strict=False)

    predict(model, test_set)
