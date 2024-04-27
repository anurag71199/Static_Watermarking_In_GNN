import torch_scatter
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import sys
# from torch_geometric.datasets import Planetoid, DBP15K, Reddit, PPI, Amazon

torch.manual_seed(1337)

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

    def conv1_output(self, x, edge_index):
        return torch.relu(self.conv1(x, edge_index))


def reset_parameters(model):
    for name, param in model.named_parameters():
        if 'weight' in name:
            init.xavier_uniform_(param)
        elif 'bias' in name:
            init.zeros_(param)

def embedding_parameter_generator(b,T,M,method="direct"):
    if method == "direct":
        two_d_tensor = torch.zeros((T, M))

        for i in range(T):
            index = torch.randint(0, M, (1,))
            two_d_tensor[i, index] = 1
        for i in range(T): 
            two_d_tensor[i, index] = 1
        return two_d_tensor
    
    if method == "diff":
        matrix = torch.zeros(T, M)
        # for i in range(T):
        #     indices = torch.randperm(M)[:2]
        #     matrix[i, indices[0]] = 1
        #     matrix[i, indices[1]] = -1
        for i in range(M):
            indices = torch.randperm(T)[:2]
            # matrix[i, indices[0]] = 1
            # matrix[i, indices[1]] = -1
            matrix[indices[0], i] = 1
            matrix[indices[1], i] = -1
        # print("X :")
        # print(matrix)
        return matrix
    
    if method == "random":
        matrix = torch.randn(T, M)
        # print("X :")
        # print(matrix)
        return matrix


def step(x):
    if x>=0:
        return 1
    return 0


class WatermarkLoss(torch.nn.Module):
    def __init__(self, X):
        super(WatermarkLoss, self).__init__()
        self.X = X  # Secret key matrix

    def forward(self, conv1, b):
        conv1_flat_ = conv1.flatten()
        y_ = torch.sigmoid(torch.matmul(self.X, conv1_flat_.t()))
        loss = 0
        for i in range(y_.size(0)):
        # Compute binary cross entropy loss for each bit
            loss += -torch.sum(b[i] * torch.log(y_[i]) + (1 - b[i]) * torch.log(1 - y_[i]))
        
        return loss
    def verify_watermark(self,b,T,X):
    
        conv1 = model.conv1_output(dataset.data.x, dataset.data.edge_index)
        conv1_flat = conv1.flatten()
        temp = torch.matmul(self.X, conv1_flat.t())
        t = list(map(step,temp))
        return t.count(1)/len(t)
        
    def step(x):
        if x>=0:
            return 1
        return 0
        

dataset = Planetoid(root='/tmp/Cora', name='Cora')
# dataset = Planetoid(root='data/PubMed', name='PubMed') 

print("Dataset : ",dataset)
method = sys.argv[1]


# Define model, optimizer, and loss function
model = GCN(input_dim=dataset.num_features, hidden_dim=16, output_dim=dataset.num_classes)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(dataset.data.x, dataset.data.edge_index)
    loss = criterion(out[dataset.data.train_mask], dataset.data.y[dataset.data.train_mask])
    loss.backward()
    optimizer.step()
    # if epoch % 10 == 0:
    #     print('Epoch:', epoch, 'Loss:', loss.item())

# Evaluate model accuracy without watermark
model.eval()
_, pred_no_watermark = model(dataset.data.x, dataset.data.edge_index).max(dim=1)
correct_no_watermark = int(pred_no_watermark[dataset.data.test_mask].eq(dataset.data.y[dataset.data.test_mask]).sum().item())
total_no_watermark = int(dataset.data.test_mask.sum())
accuracy_no_watermark = correct_no_watermark / total_no_watermark
print('Accuracy without watermark:', accuracy_no_watermark)

reset_parameters(model)

#Embedding T bit vector
T = 128
b = torch.ones(T)

#Creating embedding parameter
conv1_output_temp = model.conv1_output(dataset.data.x, dataset.data.edge_index)
M = conv1_output_temp.shape[0]*conv1_output_temp.shape[1]
X = embedding_parameter_generator(b,T,M,method)

watermark_loss = WatermarkLoss(X)

#Hyper-parameter for custom regularizer
lmd = 0.001

model.train()
for epoch in range(200):
    optimizer.zero_grad()

    out = model(dataset.data.x, dataset.data.edge_index)
    # Classification loss
    one = criterion(out[dataset.data.train_mask], dataset.data.y[dataset.data.train_mask])

    conv1_output = model.conv1_output(dataset.data.x, dataset.data.edge_index)
    two = lmd*watermark_loss(conv1_output, b)

    loss = one + two
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    if epoch % 50 == 0:
        print('Epoch:', epoch, 'Loss:', loss.item())
        print('Classification Loss:', one.item())
        print('Embedding Regularizer:', two.item())# print("parameters")

# Evaluate model accuracy with watermark
model.eval()
_, pred_with_watermark = model(dataset.data.x, dataset.data.edge_index).max(dim=1)
correct_with_watermark = int(pred_with_watermark[dataset.data.test_mask].eq(dataset.data.y[dataset.data.test_mask]).sum().item())
total_with_watermark = int(dataset.data.test_mask.sum())
accuracy_with_watermark = correct_with_watermark / total_with_watermark
print('Accuracy with watermark:', accuracy_with_watermark)


# Verify the watermark
print("watermarking accuracy : ",watermark_loss.verify_watermark(b,T,X))