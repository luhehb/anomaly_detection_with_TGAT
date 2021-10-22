import time

import dgl
from sklearn.metrics import f1_score,accuracy_score,recall_score,average_precision_score,roc_auc_score,roc_curve
from sklearn.metrics import precision_score
import  torch
from loss import loss_function,anomaly_score
import  numpy as np
import  dgl.function as fn
import torch.nn as nn

def epoch_evaluate1(args,g,model,data_center,data,radius,mask):
    model.eval()

    with torch.no_grad:
        #TODO: 节点数据不存在Labels---边label  传播到点上去
        # import dgl.function as F
        # G G.updata_all(F.copy_e('label','m'),F.max('m','label'))
        g.update_all(fn.copy_e('label','m'),fn.max('m','label'))
        labels = g.ndata['label']
        #labels= data.edata['labels']
        #loss_mask = mask.bool()&data['labels'].bool()

        #TODO： 如何由模型获取 Outputs
        outputs = model(data['g'],data['features'])

        _,scores = anomaly_score(data_center,outputs,radius,mask)
        loss,_,_= loss_function(args.nu,data_center,outputs,radius,loss_mask=None)
        labels = labels.cpu().numpy()
        scores = scores.cpu().numpy()

        threshold=0
        pred = thresholding(scores,threshold)

        auc = roc_auc_score(labels,scores)
        ap=average_precision_score(labels,scores)
        acc= accuracy_score(labels,pred)
        recall=recall_score(labels,pred)
        #precision_score= precision_score(labels,pred)
        f1=f1_score(labels,pred)
    return auc, ap, acc, recall, precision_score,f1

def epoch_evaluate(args,g,dataloader,attn,decoder, data_center,radius,device,mask=None):
    #m_ap, m_auc, m_acc =[[],[],[]]
    m_loss =[]
    m_infer_time =[]
    #TODO: 对图进行 labels 传递，得到lebals
    #labels=
    final_scores=[]
    final_labels=torch.Tensor()
    #g.update_all(fn.copy_e('label', 'm'), fn.max('m', 'label'))
    #labels = g.ndata['label']
    with torch.no_grad():
        attn.eval()
        decoder.eval()
        for batch_idx,(input_nodes,pos_graph,neg_graph,blocks,frontier,current_ts) in enumerate(dataloader):
            pos_graph = pos_graph.to(device)
            for j in range(args.n_layer):
                blocks[j] = blocks[j].to(device)

            current_ts,pos_ts,num_pos_nodes = get_current_ts(pos_graph)
            pos_graph.ndata['ts'] = current_ts
            pos_graph.update_all(fn.copy_e('label', 'm'), fn.max('m', 'label'))
            labels =pos_graph.ndata['label']
            start = time.time()
            blocks = attn.forward(blocks)
            emb = blocks[-1].dstdata['h']

            mask= ~labels.bool().squeeze()

            _,scores = anomaly_score(data_center,emb,radius,mask=mask)
            loss,_,_= loss_function(args.nu,data_center,emb,pos_graph,radius,mask=None)

            m_loss.append(loss)

            final_scores=np.concatenate((final_scores,scores.numpy()),axis=0)
            final_labels=torch.cat((final_labels,labels),dim=0)
            m_infer_time.append(start)
    #print(final_labels.numpy().shape[0])
    #print(final_scores.shape[0])
    threld =0
    pred=thresholding(final_scores,threld)
    auc=roc_auc_score(final_labels,final_scores)
    ap=average_precision_score(final_labels,final_scores)
    #acc= accuracy_score(labels,pred)
    acc=0
    print(auc,ap,acc)
    attn.train()
    decoder.train()
    return ap,auc,acc



def get_current_ts(pos_graph):
    with pos_graph.local_scope():
        pos_graph_ = dgl.add_reverse_edges(pos_graph,copy_edata=True)
        pos_graph_.update_all(fn.copy_e('timestamp','times'),fn.max('times','ts'))
        current_ts = pos_ts = pos_graph_.ndata['ts']
        num_pos_nodes = pos_graph_.num_nodes()
        return current_ts,pos_ts,num_pos_nodes
def thresholding(recon_error,threshold):
    ano_pred =np.zeros(recon_error.shape[0])
    for i in range(recon_error.shape[0]):
        if recon_error[i] > threshold:
            ano_pred[i]=1
    return ano_pred
