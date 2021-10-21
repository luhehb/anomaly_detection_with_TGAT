import torch
from torch import nn


class Decoder(nn.Module):
    def __init__(self, args, nfeat_dim):
        super().__init__()
        self.args = args
        dropout = args.dropout
        self.linkpredlayer = LinkPredLayer(nfeat_dim * 2, 1, dropout)

    def linkpred(self, block_outputs, pos_graph, neg_graph):
        # with neg_graph.local_scope():
        #     neg_graph.ndata['feat'] = block_outputs
        #     neg_graph.apply_edges(linkpred_concat)
        #     neg_emb = neg_graph.edata['emb']
        with pos_graph.local_scope():
            pos_graph.ndata['feat'] = block_outputs
            pos_graph.apply_edges(linkpred_concat)
            pos_emb = pos_graph.edata['emb']
        logits = self.linkpredlayer(pos_emb)
        labels = torch.zeros_like(logits)
        labels[:pos_emb.shape[0]] = 1
        return logits, labels

    def forward(self, block_outputs, pos_graph, neg_graph):
        logits, labels = self.linkpred(block_outputs, pos_graph, neg_graph)
        return logits.squeeze(), labels.squeeze()

def linkpred_concat(edges):
    return {'emb': torch.cat([edges.src['feat'], edges.dst['feat']], 1)}

class LinkPredLayer(nn.Module):
    def __init__(self, in_dim, class_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.BatchNorm1d(in_dim)
        hidden_dim = in_dim // 2
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, class_dim)
        self.act = nn.ReLU()

        self.dropout = nn.Dropout(p=dropout, inplace=True)

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, h):
        h = self.dropout(h)
        h = self.fc1(h)
        h = self.act(h)
        h = self.fc2(h)
        return h
