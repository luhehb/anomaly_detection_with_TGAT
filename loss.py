import torch
import numpy as np

def loss_function(nu,data_center,outputs, radius=0,mask=None):
    #mask=~graph.ndata['label'].bool().squeeze()
    #print(mask)
    dist, scores = anomaly_score(data_center,outputs,radius,mask)
    loss = radius**2+(1/nu)*torch.mean(torch.max(torch.zeros_like(scores),scores))
    return loss, dist, scores


def anomaly_score(data_center,outputs,radius=0,mask=None):
    if mask==None:
        dist = torch.sum((outputs-data_center)**2,dim=1)
    else:
        dist=torch.sum((outputs[mask]-data_center)**2,dim=1)

    scores = dist-radius**2
    return dist, scores


def init_center(args,emb,mask):
    if args.gpu<0:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:%d'%args.gpu)


def init_center(args,blocks,model,eps=0.001):
    """
    Initialize hypersphere center c
    as the mean from an initial forward pass on the data.
    """
    if args.gpu<0:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:%d' % args.gpu)
    n_samples=0

    #TODO 输出向量维度需要进一步确认

    c=torch.zeros(args.n_hidden,device = device)
    model.eval()
    with torch.no_grad():
        blocks = model(blocks)
        outputs=blocks[-1].dstdata['h']
        n_samples = outputs.shape[0]
        c=torch.sum(outputs,dim=0)
    c/=n_samples
    """
    if c is too close to 0, set to +-eps
    Reason： a zero unit can be trivially matched with zero weights
    """
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps

    return c

def get_radius(dist: torch.Tensor,nu: float):
    """
    Optimally solve for radius R via the (1-nu)-quantile of distances
    """
    radius=np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)
    # if radius<0.1:
    #     radius=0.1
    return radius