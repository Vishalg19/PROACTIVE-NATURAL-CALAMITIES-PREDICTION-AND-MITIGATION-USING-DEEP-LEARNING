# Import necessary libraries
import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv

# Load earthquake data from CSV
earthquake_data = pd.read_csv('earthquake_data.csv')

# Create a graph using networkx
G = nx.Graph()
threshold=7.0
# Add nodes with attributes (example: longitude, latitude, magnitude)
for index, row in earthquake_data.iterrows():
    G.add_node(index, lon=row['longitude'], lat=row['latitude'], mag=row['magnitude'])

# Add edges based on some criteria (example: geographical proximity)
# You might have a more sophisticated approach based on your domain knowledge
    
for i in range(len(earthquake_data)):
    for j in range(i+1, len(earthquake_data)):
        distance = ((G.nodes[i]['lon'] - G.nodes[j]['lon'])**2 + 
                    (G.nodes[i]['lat'] - G.nodes[j]['lat'])**2)**0.5
        if distance < threshold:# Define your threshold based on proximity
            G.add_edge(i, j)

# Convert networkx graph to PyTorch Geometric Data
edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
x = torch.tensor(earthquake_data[['longitude', 'latitude', 'magnitude']].values, dtype=torch.float)
y = torch.tensor(earthquake_data['dmin'].values, dtype=torch.float)  # Add labels if available

data = Data(x=x, edge_index=edge_index, y=y)

# Define a simple GCN model
class GCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Instantiate the model, loss function, and optimizer
model = GCNModel(input_dim=3, hidden_dim=64, output_dim=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model
epochs=50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, data.y.view(-1, 1))
    loss.backward()
    optimizer.step()

# Evaluate the model (you might want to use a separate test set)
model.eval()
with torch.no_grad():
    test_output = model(data)
    # Evaluate performance metrics based on your task
