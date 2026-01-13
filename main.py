# SECTION: Imports

# Import the core functionality of PyTorch
import torch

# Brings the nn (neural network) module into scope.
# This contains tools for working with neural networks.
import torch.nn as nn

# Brings the optim module into scope. 
# This module contains optimization algorithms that allow the model to
# update its parameters and make better predictions.
import torch.optim as optim

#______________________________________________________________________________

# SECTION: Result Reproducibility

# This will make all CPU-based random operations deterministic.
# It ensures that the results are reproducible and consistent every time.
torch.manual_seed(42)

#______________________________________________________________________________

# SECTION: The Machine Learning Pipeline

#______________________________________________________________________________

# SUB_SECTION: `Step 1 - Data Ingestion` and `Step 2 - Data Preparation`

# The delivery records data used here has already been cleaned.

# | Distance (miles) | Delivery Time (minutes) |
# |------------------|-------------------------|
# | 1.0              | 6.96                    |
# | 2.0              | 12.11                   |
# | 3.0              | 16.77                   |
# | 4.0              | 22.21                   |


# A tensor is created for the distances (which will be an input for the model)
distances = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)

# Another tensor is created for the delivery times 
# (which will be an output for the model)
times = torch.tensor([[6.96], [12.11], [16.77], [22.21]], dtype=torch.float32)

#______________________________________________________________________________

# SUB_SECTION: Step 3 - Model Building

# The ML model used here will be a linear equation 
# that will take one input (the distance in miles), 
# and it will use this to produce one output (the delivery time in minutes)
model = nn.Sequential(nn.Linear(1, 1))

# This creates a loss_function
# MSE (Mean squared error loss) will be used.
# This will measure how far the model's predictions 
# are from the actual values in the dataset.
loss_function = nn.MSELoss()

# This creates the optimizer.
# SDG (Stochastic gradient descent)
# This is an algorithm that figures out which direction to adjust 
# the `W` and `b` parameters of the model (Weight and bias)
# in order to reduce that error.

# `lr` is the learning rate. 
# This controls how much the model will try to adjust itself to make its 
# predictions more accurate.

# If `lr` is too large, the model may overshoot the best values.
# If `lr` is too small, the machine learning process may take too long.
optimizer = optim.SGD(model.parameters(), lr=0.01)

#______________________________________________________________________________

def main():
    pass

if __name__ == "__main__":
    main()
