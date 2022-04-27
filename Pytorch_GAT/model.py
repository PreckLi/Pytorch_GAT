import torch
from torch import nn
from layers import GraphAttentionLayer
from torch.nn import functional as F


class GAT(nn.Module):
    def __init__(self, in_size, hid_size, num_class, dropout, alpha, num_head):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = list()
        for _ in range(num_head):
            self.attentions.append(GraphAttentionLayer(in_size, hid_size, dropout, alpha, concat=True))

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(hid_size * num_head, num_class, dropout, alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x_list = list()
        for att in self.attentions:
            x_list.append(att(x, adj))
        x = torch.cat(x_list, dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)
