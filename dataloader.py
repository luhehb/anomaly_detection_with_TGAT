import dgl
import torch
import random
import dgl.function as fn

class TemporalEdgeCollator(dgl.dataloading.EdgeCollator):
    '''
    整理边
    '''
    def __init__(self,args,g,eids,block_sampler,g_sampling=None,exclude=None,
                reverse_eids=None,reverse_etypes=None,negative_sampler=None):
        super(TemporalEdgeCollator,self).__init__(g,eids,block_sampler,
                                                 g_sampling,exclude,reverse_eids,reverse_etypes,negative_sampler)
        self.args=args

    # items是采到的batch_size个edges
    def collate(self,items):
        # only sample edges before current timestamp
        current_ts=self.g.edata['timestamp'][items[0]]
        # 当前block的最后一个ts，给下面的MultiLayerTemporalNeighborSampler定义的
        self.block_sampler.ts=current_ts
        neg_pair_graph=None
        if self.negative_sampler is None:
            input_nodes,pair_graph,blocks=self._collate(items)
        else:
            input_nodes,pair_graph,neg_pair_graph,blocks=self._collate_with_negative_sampling(items)
        if self.args.n_layer>1:
            # 为什么要叠加？
            self.block_sampler.frontiers[0].add_edges(*self.block_sampler.frontiers[1].edges())
        # 上行代码好像把边全部加到frontier[0]里面了
        frontier=dgl.reverse(self.block_sampler.frontiers[0])
        # frontier：包含原图所有节点，但是只有在此层中有message passing的edge才被包含
        return input_nodes, pair_graph, neg_pair_graph, blocks, frontier, current_ts


class ValSampler(dgl.dataloading.BlockSampler):
    '''
    对边采样，返回block的frontier。
    '''

    def __init__(self, args,fanouts, replace=False, return_eids=False):
        super().__init__(len(fanouts), return_eids)

        self.fanouts = fanouts  # fanout应该是每层要采样的节点个数
        self.replace = replace
        self.ts = 0
        self.args = args
        self.frontiers = [None for _ in range(len(fanouts))]

    def sample_frontier(self, block_id, g, seed_nodes):
        fanout = self.fanouts[block_id]
        # List of neighbors to sample per edge type for each GNN layer, starting from the first layer.
        g = dgl.in_subgraph(g, seed_nodes)  # 有所有点，以及seed node的入边
        g.remove_edges(torch.where(g.edata['timestamp'] > self.ts)[0])  # 去掉ts之后的edge
        # torch.where(条件，符合条件设置为；不符合条件设置为)
        frontier = g
        self.frontiers[block_id] = frontier  # 每层blocks采样的点存在frontier里面
        return frontier

class TrainSampler(dgl.dataloading.BlockSampler):
    '''
    对边采样，返回block的frontier。
    '''

    def __init__(self, args,fanouts, replace=False, return_eids=False):
        super().__init__(args.n_layer, return_eids)

        self.fanouts = fanouts  # fanout应该是每层要采样的节点个数
        self.replace = replace
        self.ts = 0
        self.args = args
        self.frontiers = [None for _ in range(args.n_layer)]

    def sample_prob(self, edges):
        timespan = edges.dst['sample_time'] - edges.data['timestamp']
        # zeros=torch.zeros_like(timespan)
        # print(timespan.size(),zeros.size())
        #mask = timespan < 0
        #prob = self.softmax(timespan)
        #prob[mask] = 0  # 把ts大于当前的
        # print(prob)
        return {'timespan': timespan}

    def sample_time(self, edges):
        return {'st': edges.data['timestamp']}

    def sample_frontier(self, block_id, g, seed_nodes):
        fanout = self.fanouts[block_id] if self.fanouts is not None else None
        # List of neighbors to sample per edge type for each GNN layer, starting from the first layer.
        g = dgl.in_subgraph(g, seed_nodes)  # 只包含seed_node的g（不考虑邻居吗？）
        g.remove_edges(torch.where(g.edata['timestamp'] > self.ts)[0])  # 去掉ts之后的edge
        """
        去掉Label ==1 的边的 节点
        """
        # anomalyEdges=torch.where(g.edata['label']==1)
        # srcNodeIds,dstNodeIds=g.find_edges(anomalyEdges[0].data)
        # g.remove_nodes(srcNodeIds)
        # g.remove_nodes(dstNodeIds)

        # if block_id != self.args.n_layer - 1:  # 只要不是最后一层
        #     # 当前层dst的sampletime等于里层src的sample_time
        #     g.dstdata['sample_time'] = self.frontiers[block_id + 1].srcdata['sample_time']
        #     g.apply_edges(self.sample_prob)
        #     g.remove_edges(torch.where(g.edata['timespan'] < 0)[0])

        # #用翻转更新g的src的sample time
        # g_re=dgl.reverse(g,copy_edata=True,copy_ndata=True)#翻转
        # g_re.update_all(self.sample_time,fn.max('st','sample_time'))
        # g=dgl.reverse(g_re,copy_edata=True,copy_ndata=True)

        #g.update_all(self.sample_time, fn.max('st', 'sample_time'))
        #print(g.srcdata['sample_time'])
        # torch.where(条件，符合条件设置为；不符合条件设置为)
        if fanout is None:  # 不采样就取全图
            frontier = g
        # else:
        #     if block_id == self.args.n_layer - 1:  # 如果是最外层，就按照timestamp采样
        #         frontier = dgl.sampling.select_topk(g, fanout, 'timestamp', seed_nodes)
        #     else:
        #         frontier = dgl.sampling.select_topk(g, fanout, 'timespan', seed_nodes,
        #                                             ascending=True)  # 选timestamp最大的【fanout】个点
        else:
            if self.args.uniform:
                frontier = dgl.sampling.sample_neighbors(g, seed_nodes, fanout)
            else:
                frontier = dgl.sampling.select_topk(g, fanout, 'timestamp', seed_nodes)
        # 每层blocks采样的点存在frontier里面
        self.frontiers[block_id] = frontier
        # print(type(frontier))
        return frontier

def dataloader(args,g):
    '''
    原始边数量为何为  边数量为何要 一半
    '''
    origin_num_edges = g.num_edges() // 2

    train_eid = torch.arange(0, int(0.7 * origin_num_edges))
    val_eid = torch.arange(int(0.7 * origin_num_edges), int(0.85 * origin_num_edges))
    test_eid = torch.arange(int(0.85 * origin_num_edges), origin_num_edges)

    # reverse_eids = torch.cat([torch.arange(origin_num_edges, 2 * origin_num_edges), torch.arange(0, origin_num_edges)])
    exclude, reverse_eids = None, None
    # 为啥要生成负样本
    #negative_sampler = dgl.dataloading.negative_sampler.Uniform(1)
    negative_sampler=None
    # 语义？
    fan_out = [args.n_degree for _ in range(args.n_layer)]

    train_sampler = TrainSampler(args, fanouts=fan_out, return_eids=False)
    val_sampler=ValSampler(args, fanouts=fan_out, return_eids=False)
    train_collator = TemporalEdgeCollator(args,g, train_eid, train_sampler, exclude=exclude, reverse_eids=reverse_eids,
                                          negative_sampler=negative_sampler)

    train_loader = torch.utils.data.DataLoader(
        train_collator.dataset, collate_fn=train_collator.collate,
        batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.n_worker)
    val_collator = TemporalEdgeCollator(args,g, val_eid, val_sampler, exclude=exclude, reverse_eids=reverse_eids,
                                        negative_sampler=negative_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_collator.dataset, collate_fn=val_collator.collate,
        batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.n_worker)
    test_collator = TemporalEdgeCollator(args,g, test_eid, val_sampler, exclude=exclude, reverse_eids=reverse_eids,
                                         negative_sampler=negative_sampler)
    test_loader = torch.utils.data.DataLoader(
        test_collator.dataset, collate_fn=test_collator.collate,
        batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.n_worker)
    return train_loader, val_loader, test_loader, val_eid.shape[0], test_eid.shape[0]