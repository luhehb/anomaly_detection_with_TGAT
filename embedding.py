import torch
import torch.nn as nn

def init_embedding(g):
    g.ndata['h']=g.ndata['feat']

class ATTN(nn.Module):
    def __init__(self,args,time_encoder):
        super().__init__()
        self.n_layer = args.n_layer
        self.h_dimension = args.emb_dimension
        self.attnlayers = nn.ModuleList()
        self.mergelayers=nn.ModuleList()
        self.edge_feat_dim=args.edge_feat_dim
        self.n_head = args.n_head
        self.time_dim = args.time_dimension
        self.node_feat_dim = args.node_feat_dim
        self.dropout = nn.Dropout(p=args.dropout, inplace=True)
        self.args=args
        self.time_encoder=time_encoder
        self.device='cuda:0' if torch.cuda.is_available() else 'cpu'

        self.query_dim = self.node_feat_dim + self.time_dim  # 仅仅只是维
        self.key_dim = self.node_feat_dim + self.time_dim + self.edge_feat_dim
        for i in range(0, self.n_layer):
            self.attnlayers.append(nn.MultiheadAttention(embed_dim=self.query_dim,
                                                       kdim=self.key_dim,
                                                       vdim=self.key_dim,
                                                       num_heads=self.n_head,
                                                       dropout=args.dropout).to(self.device))
            self.mergelayers.append(MergeLayer(self.query_dim, self.node_feat_dim, self.node_feat_dim, self.h_dimension).to(self.device))

    def C_compute(self,edges):
        te_C = self.time_encoder(edges.data['timestamp'] - edges.src['last_update'])
        C = torch.cat([edges.src['h'], edges.data['feat'], te_C], dim=1)
       # print(C)
        return {'C': C}

    def h_compute(self,nodes):
        C=nodes.mailbox['C']
        C=C.permute([1,0,2])#convert to [num_node,num_neighbor,feat_dim]
        key = C.to(self.device)

        te_q=self.time_encoder(torch.zeros(nodes.batch_size()).to(self.device))
        query = torch.cat([nodes.data['h'], te_q],dim=1).unsqueeze(dim=0)

        h_before,_= self.attnlayers[self.l](query=query, key=key, value=key)
        h_before=h_before.squeeze(0)

        h= self.mergelayers[self.l](nodes.data['h'], h_before)
        return {'h':h}

    def forward(self, blocks): # x是h
        for l in range(self.n_layer):
            self.l=l
            blocks[l].update_all(self.C_compute,self.h_compute)
            if l!=self.n_layer-1:#如果不是最后一层，那么上一层的dst数据要同步到下一层的src
                blocks[l+1].srcdata['h']=blocks[l].dstdata['h']
        return blocks

class MergeLayer(torch.nn.Module):
    '''(dim1+dim2)->dim3->dim4'''

    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        h = self.act(self.fc1(x))
        return self.fc2(h)

