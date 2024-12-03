import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Fetch dataset
connect_4 = fetch_ucirepo(id=26)  # Make sure this works properly

# Data (as pandas dataframes)
X = connect_4.data.features
y = connect_4.data.targets

# Manually map 'x', 'o', 'b' to numerical values
# 'b' -> 0 (empty), 'x' -> 1 (Player 1), 'o' -> 2 (Player 2)
X_mapped = X.replace({'b': 0, 'x': 1, 'o': 2}).values

# Add player ID to each row
player_ids = []
for row in X.values:
    count_x = np.sum(row == 'x')
    count_o = np.sum(row == 'o')
    
    # Determine the current player based on the number of tokens
    if count_x == count_o:
        player_ids.append(1)  # Player 1's turn
    else:
        player_ids.append(2)  # Player 2's turn

# Convert player_ids to numpy array and concatenate to X_mapped
player_ids = np.array(player_ids).reshape(-1, 1)
X_with_player = np.hstack((X_mapped, player_ids))

# Encode outcomes ('win', 'loss', 'draw')
y_encoder = LabelEncoder()
y_encoded = y_encoder.fit_transform(y)  # Convert 'win', 'loss', 'draw' into numerical labels

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_with_player, y_encoded, test_size=0.1, random_state=42)

# Convert to torch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)


# Define the model for predicting the game outcome considering the player ID
class Connect4OutcomePredictor(nn.Module):
    def __init__(self):
        super(Connect4OutcomePredictor, self).__init__()
        self.fc1 = nn.Linear(43, 128)  # 42 board positions + 1 player ID
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)  # Output layer with 3 classes (win, loss, draw)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model, loss function, and optimizer
model = Connect4OutcomePredictor()
criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss works well for classification tasks
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 400
for epoch in range(num_epochs):
    # Forward pass
    model.train()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Calculate and print validation loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_tensor)
            val_loss = criterion(val_outputs, y_test_tensor)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

# Evaluating the model
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
    print(f'Accuracy on test set: {accuracy * 100:.2f}%')

# Write weights to a text file
fc1_weights = model.fc1.weight.data.numpy()

# Format the weights to look like a Python list of lists
with open('formatted_model_weights_fc1.txt', 'w') as f:
    f.write("[\n")  # Start of the list
    for i, row in enumerate(fc1_weights):
        row_str = ", ".join([f"{value:.6f}" for value in row])
        if i < len(fc1_weights) - 1:
            f.write(f"  [{row_str}],\n")  # Comma at the end of each row except the last
        else:
            f.write(f"  [{row_str}]\n")  # No comma for the last row
    f.write("]\n")  # End of the list
fc2_weights = model.fc2.weight.data.numpy()

# Format the weights to look like a Python list of lists
with open('formatted_model_weights_fc2.txt', 'w') as f:
    f.write("[\n")  # Start of the list
    for i, row in enumerate(fc2_weights):
        row_str = ", ".join([f"{value:.6f}" for value in row])
        if i < len(fc2_weights) - 1:
            f.write(f"  [{row_str}],\n")  # Comma at the end of each row except the last
        else:
            f.write(f"  [{row_str}]\n")  # No comma for the last row
    f.write("]\n")  # End of the list
    
fc3_weights = model.fc3.weight.data.numpy()

# Format the weights to look like a Python list of lists
with open('formatted_model_weights_fc3.txt', 'w') as f:
    f.write("[\n")  # Start of the list
    for i, row in enumerate(fc3_weights):
        row_str = ", ".join([f"{value:.6f}" for value in row])
        if i < len(fc3_weights) - 1:
            f.write(f"  [{row_str}],\n")  # Comma at the end of each row except the last
        else:
            f.write(f"  [{row_str}]\n")  # No comma for the last row
    f.write("]\n")  # End of the list
# Corrected code to save fc1, fc2, and fc3 biases

# Save fc1 bias
fc1_bias = model.fc1.bias.data.numpy()
with open('formatted_model_bias_fc1.txt', 'w') as f:
    # Format as a simple list
    row_str = ", ".join([f"{value:.6f}" for value in fc1_bias])
    f.write(f"[{row_str}]\n")

# Save fc2 bias
fc2_bias = model.fc2.bias.data.numpy()
with open('formatted_model_bias_fc2.txt', 'w') as f:
    row_str = ", ".join([f"{value:.6f}" for value in fc2_bias])
    f.write(f"[{row_str}]\n")

# Save fc3 bias
fc3_bias = model.fc3.bias.data.numpy()
with open('formatted_model_bias_fc3.txt', 'w') as f:
    row_str = ", ".join([f"{value:.6f}" for value in fc3_bias])
    f.write(f"[{row_str}]\n")
