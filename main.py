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

# SUB_SECTION: Step 4 - Training

# Each round of training is called an `epoch`
# The model will go through the data 500 times to figure out a pattern
# that will allow it to make predictions for the delivery times of distances
# that are not in the data.
for epoch in range(500):

    # 0. Clears all of the calculated values from the previous epoch.
    # Without this, PyTorch would accumulate adjustments, 
    # which could break the learning process.
    optimizer.zero_grad()

    # 1. Make a prediction
    # The syntax is `outputs = model(inputs)`
    # Performs the "forward pass", 
    # where the model makes predictions based on the input distances.
    outputs = model(distances)

    # 2. Calculates how wrong the predicted outputs are by comparing them 
    # to the actual delivery times in the data.
    # The syntax is `loss_function(predictions, targets)`
    loss = loss_function(outputs, times)

    # 3. This is where the model adjusts its parameters 
    # to improve its predictions
    # The technical term is Back Propagation.
    loss.backward()

    # 4. Update the model
    optimizer.step()

#______________________________________________________________________________

# SUB_SECTION: Step 5 - Evaluation

distance_to_predict = 7.0

distance_to_predict = 7.0

# This tells PyTorch that I am not training the model anymore,
# and that I'm doing inference (Making a prediction for a data point that I
# don't have)
with torch.no_grad():
    # Convert the Python variable into a 2D PyTorch tensor that the model expects
    new_distance = torch.tensor([[distance_to_predict]], dtype=torch.float32)
    
    # Pass the new data to the trained model to get a prediction
    predicted_time = model(new_distance)
    
    # Use .item() to extract the scalar value from the tensor for printing
    print()
    print(f"Prediction for a {distance_to_predict}-mile delivery: {predicted_time.item():.1f} minutes")

    # Use the scalar value in a conditional statement to make the final decision
    if predicted_time.item() > 30:
        print()
        print("Decision: ❌ Reject the delivery.")
    else:
        print()
        print("Decision: ✅ Accept the delivery.")

#______________________________________________________________________________

def main():
    pass

if __name__ == "__main__":
    main()
