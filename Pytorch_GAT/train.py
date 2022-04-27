import torch
from model import GAT
from torch_geometric.datasets import Planetoid
from torch import optim
from torch.nn import functional as F
from utils import get_adjacency_matrix

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
coradataset=Planetoid('../GNN_learning/tmp/Cora','Cora')
data=coradataset[0].to(device)
adj=get_adjacency_matrix(data.edge_index,data.num_nodes).to(device)

def train(epoch):
    GATmodel=GAT(in_size=coradataset.num_node_features,hid_size=16,num_class=coradataset.num_classes,dropout=0.5,alpha=0.2,num_head=8).to(device)
    optimizer=optim.Adam(GATmodel.parameters(),lr=0.01,weight_decay=0.0005)
    GATmodel.train()

    for i in range(epoch):
        optimizer.zero_grad()
        output=GATmodel.forward(data.x,adj)
        loss=F.nll_loss(output[data.train_mask],data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if (i+1)%20==0:
            print('epoch:{},loss:{}'.format(i+1,loss.data))

    return GATmodel


def evaluate(GATmodel:GAT):
    GATmodel.eval()
    pred=GATmodel.forward(data.x,adj).argmax(dim=1)
    correct=(pred[data.test_mask]==data.y[data.test_mask]).sum()
    acc=int(correct)/int(data.test_mask.sum())
    print("accuracy of GAT:{}".format(acc))

pass