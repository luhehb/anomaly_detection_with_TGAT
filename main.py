from args import get_args
from dataloader import dataloader
from dgl.data.utils import load_graphs
import torch
import dgl
import  numpy as np
from embeding import init_embedding,ATTN
from time_encode import TimeEncode
from decoder import  Decoder
#from val_eval import get_current_ts,eval_epoch
from loss import  loss_function,get_radius,init_center
from evaluate import epoch_evaluate,get_current_ts
import dgl.function as fn

if __name__ == '__main__':
    args = get_args()

    g = load_graphs("wikipedia.bin")[0][0]

    # 边特征维度
    efeat_dim = g.edata['feat'].shape[1]

    #设置是否支持 cuda
    device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    #初始化节点labels
    g.update_all(fn.copy_e('label', 'm'), fn.max('m', 'label'))
    labels = g.ndata['label']
    #print(labels)
    # 初始化节点feat----图节点无特征数据
    node_feature = torch.zeros((g.number_of_nodes(),args.emb_dimension))
    g.ndata['feat'] = node_feature
    # 处理数据--为啥子需要这些数据，数据是如何得到的？
    train_loader, val_loader, test_loader, val_enum, test_enum = dataloader(args,g)

    t0=torch.zeros(g.number_of_nodes())

    time_encoder = TimeEncode(args.time_dimension).to(device)

    emb_updater = ATTN(args, time_encoder).to(device)

    decoder = Decoder(args, args.emb_dimension)

    #loss_fcn = torch.nn.BCEWithLogitsLoss().to(device)
    #loss_fcn=loss_function().to(device)
    #建立矩阵存储loss-------到底是多少呢？

    #初始化球心
    data_centerbase = torch.zeros(args.emb_dimension)
    #初始化半径
    radius=torch.tensor(0)
    #loss_fcn = loss_function(args.nu,data_center,r).to(device)
    # 做梯度下降，
    optimizer = torch.optim.Adam(list(decoder.parameters())+list(emb_updater.parameters()), lr=args.lr,
                                 weight_decay=args.weight_decay)
    #记录所有的 epoch 数据的均值
    epoch_data_center = torch.tensor([])
    all_epoch_mean = []

    data_center =torch.zeros(args.emb_dimension).to(device)
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

        batch_data_center_list=[]
        # epoch_data_center !=null---> batch_data_center 与

        for batch_id, (input_nodes, pos_graph, neg_graph, blocks, frontier, current_ts) in enumerate(train_loader):
            pos_graph = pos_graph.to(device)
           
            for j in range(args.n_layer):
                blocks[j] = blocks[j].to(device)

            current_ts, pos_ts, num_pos_nodes = get_current_ts(pos_graph)
            pos_graph.ndata['ts'] = current_ts
            #print(pos_graph.ndata['_ID'].shape[0])
            #print(pos_graph.ndata['label'].shape[0])

            # 更新emb
            blocks=emb_updater.forward(blocks)

            # 获得emb
            emb=blocks[-1].dstdata['h']

            #重解球心的尝试
            #data_center = init_center(emb)


            #print(blocks[0].num_src_nodes)
            #print(emb.shape[0])
            mask = ~pos_graph.ndata['label'].bool().squeeze()
            """
            通过emb 来获取球心
            """
            # 计算当前数据的均值
            size = emb.shape[0]
            cur_data_center = torch.mean(emb,dim=0)
            batch_data_center_list.append(cur_data_center)
            #batch_data_center = torch.cat((batch_data_center,cur_data_center))
            # 如果 epoch 为空 则 datacenter 为当前的 数据中心


            loss,dist,scores = loss_function(args.nu,data_center,emb,radius,mask)
            arr_loss.append(loss.item())

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
            print(np.mean(arr_loss))
            #更新last_update
            with torch.no_grad():
                g.ndata['last_update'][pos_graph.ndata[dgl.NID][:num_pos_nodes]] = pos_ts.to('cpu')
        

        # 评估验证集
        # val_ap, val_auc, val_acc, val_loss ,time_c= eval_epoch(args,g, val_loader, emb_updater, decoder,
        #                                                 loss_fcn, device)#评估验证集
        # print("epoch:%d,loss:%f,ap:%f,time_consume:%f" % (i, val_loss, val_ap, time_c))

        ap,auc,acc = epoch_evaluate(args,g,val_loader,emb_updater,decoder,data_center,radius,device,mask=None)
        print("epoch:%d,auc:%f,ap:%f,acc:%f"%(i,auc,ap,acc))
        #更新data_center
        #with torch.no_grad():
        # final_batch_data_center = torch.stack(batch_data_center_list,0)
        # #print(final_batch_data_center)
        # #size= final_batch_data_center.shape[0]
        # #print(size)
        # single_epoch_data_center= torch.mean(final_batch_data_center,dim=0)
        # #print(single_epoch_data_center)
        # all_epoch_mean.append(single_epoch_data_center)
        # epoch_data_center_list= torch.stack(all_epoch_mean,0)
        # #print(epoch_data_center_list)
        # data_center=(1/(i+1))*torch.sum(epoch_data_center_list,dim=0)
            #print(data_center)



