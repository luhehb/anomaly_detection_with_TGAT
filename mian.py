from args import get_args
from dataloader import dataloader
from dgl.data.utils import load_graphs
import torch
import dgl
import  numpy as np
from embeding import init_embedding,ATTN
from time_encode import TimeEncode
from decoder import  Decoder
from val_eval import get_current_ts,eval_epoch
from loss import  loss_function,get_radius,init_center
from evaluate import epoch_evaluate

if __name__ == '__main__':
    args = get_args()

    g = load_graphs(f"D:\PythonRelation\Dataset\wiki.dgl")[0][0]

    # 边特征维度
    efeat_dim = g.edata['feat'].shape[1]

    #设置是否支持 cuda
    device='cuda:0' if torch.cuda.is_available() else 'cpu'

    # 初始化节点feat----图节点无特征数据
    node_feature = torch.zeros((g.number_of_nodes(),args.emb_dimension))
    g.ndata['feat'] = node_feature
    # 处理数据--为啥子需要这些数据，数据是如何得到的？
    train_loader, val_loader, test_loader, val_enum, test_enum = dataloader(args,g)

    t0=torch.zeros(g.number_of_nodes())

    time_encoder = TimeEncode(args.time_dimension).to(device)

    emb_updater = ATTN(args, time_encoder).to(device)

    decoder = Decoder(args, 100)

    #loss_fcn = torch.nn.BCEWithLogitsLoss().to(device)
    #loss_fcn=loss_function().to(device)
    #建立矩阵存储loss-------到底是多少呢？

    #初始化球心
    data_center = torch.zeros(100)
    #初始化半径
    radius=torch.tensor(0)
    #loss_fcn = loss_function(args.nu,data_center,r).to(device)
    # 做梯度下降，
    optimizer = torch.optim.Adam(list(decoder.parameters())+list(emb_updater.parameters()), lr=args.lr,
                                 weight_decay=args.weight_decay)


    #开始训练
    for i in range(10):
        # 初始化emb
        arr_loss = []
        init_embedding(g)
        # 最初的last updated
        g.ndata['last_update'] = t0
        g.ndata['sample_time'] = t0

        decoder.train()
        time_encoder.train()
        emb_updater.train()

        for batch_id, (input_nodes, pos_graph, neg_graph, blocks, frontier, current_ts) in enumerate(train_loader):
            pos_graph = pos_graph.to(device)
           # neg_graph = neg_graph.to(device)
            for j in range(args.n_layer):
                blocks[j] = blocks[j].to(device)

            current_ts, pos_ts, num_pos_nodes = get_current_ts(pos_graph)
            pos_graph.ndata['ts'] = current_ts

            # 更新emb
            blocks=emb_updater.forward(blocks)

            # 获得emb
            emb=blocks[-1].dstdata['h']

            loss,dist,scores = loss_function(args.nu,data_center,emb,radius,mask=None)
            arr_loss.append(loss.item())
           # print(scores.detach.numpy())
            #print(scores.data.numpy())
            #print(arr_loss)

            #print(emb.shape[1])
            # # 训练
            # logits, labels = decoder(emb, pos_graph, neg_graph)
            #
            # '''
            # loss = loss_fcn(logits)
            # '''
            #
            # loss = loss_fcn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            radius.data = torch.tensor(get_radius(dist, args.nu), device=device)

            #更新last_update
            with torch.no_grad():
                g.ndata['last_update'][pos_graph.ndata[dgl.NID][:num_pos_nodes]] = pos_ts.to('cpu')
        # 评估验证集
        # val_ap, val_auc, val_acc, val_loss ,time_c= eval_epoch(args,g, val_loader, emb_updater, decoder,
        #                                                 loss_fcn, device)#评估验证集
        # print("epoch:%d,loss:%f,ap:%f,time_consume:%f" % (i, val_loss, val_ap, time_c))
        print(arr_loss)
        ap,auc,acc = epoch_evaluate(args,g,val_loader,emb_updater,decoder,data_center,radius,device,mask=None)
        print("epoch:%d,auc:%f,ap:%f,acc:%f"%(i,auc,ap,acc))