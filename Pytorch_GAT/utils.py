import torch

def get_adjacency_matrix(edge_index,num_node):
    adjacency=[[0]*num_node for _ in range(num_node)]
    x = edge_index[0]
    y = edge_index[1]
    for i, j in zip(x, y):
        adjacency[i][j]=1
        adjacency[j][i]=1
    adjacency=torch.as_tensor(adjacency,dtype=torch.float32)
    return adjacency