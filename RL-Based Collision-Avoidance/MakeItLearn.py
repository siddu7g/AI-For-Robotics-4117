import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from PreProcessing import PreprocessData
import os

# --------------------------- PARAMETERS ---------------------------
torch.manual_seed(10)
np.random.seed(10)
InputSize = 6
batch_size = 1
NumClasses = 1
NumEpochs = 20
HiddenSize = 10

# Create folder to save models if not exists
os.makedirs('./SavedNets', exist_ok=True)

# --------------------------- MODEL DEFINITION ---------------------------
class Net(nn.Module):
    def __init__(self, InputSize, NumClasses):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(InputSize, HiddenSize)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(HiddenSize, NumClasses)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# --------------------------- LOSS FUNCTION ---------------------------
criterion = nn.MSELoss()

# --------------------------- LEARNING RATES ---------------------------
learning_rates = [1e-6, 1e-4, 1e-2]
loss_records = {}

# --------------------------- TRAINING LOOP ---------------------------
if __name__ == "__main__":
    for lr in learning_rates:
        print(f"\n========== Training with lr = {lr} ==========")

        # Reload fresh data for each learning rate
        TrainSize, SensorNNData, SensorNNLabels = PreprocessData()

        # Create a new model and optimizer for each learning rate
        net = Net(InputSize, NumClasses)
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)

        epoch_losses = []

        for epoch in range(NumEpochs):
            total_loss = 0.0

            for i in range(TrainSize):
                input_values = Variable(SensorNNData[i])
                labels = Variable(SensorNNLabels[i])

                optimizer.zero_grad()
                outputs = net(input_values)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / TrainSize
            epoch_losses.append(avg_loss)
            print(f"Epoch [{epoch+1}/{NumEpochs}]  Loss: {avg_loss:.6f}")

        # Save losses for this learning rate
        loss_records[lr] = epoch_losses

        # Save the trained model
        model_path = f'./SavedNets/NNBot_lr{lr}.pkl'
        torch.save(net.state_dict(), model_path)
        print(f"Model saved at {model_path}")

    # --------------------------- PLOTTING ---------------------------
    print("\nRecorded learning rates:", loss_records.keys())

    plt.clf()
    plt.figure(figsize=(8, 6))
    for lr, losses in loss_records.items():
        plt.plot(range(1, NumEpochs + 1), losses, label=f'lr = {lr}', linewidth=2)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Loss vs Epoch for Different Learning Rates', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

'''

import torch
import torch.nn as nn
from torch.autograd import Variable
import pdb
import numpy as np
from PreProcessing import PreprocessData
import matplotlib.pyplot as plt


# Set Seeds For Randomness
torch.manual_seed(10)
np.random.seed(10)
InputSize = 6  # Input Size
batch_size = 1  # Batch Size Of Neural Network
NumClasses = 1  # Output Size

############################################# FOR STUDENTS #####################################

NumEpochs = 10
HiddenSize = 10

# Create The Neural Network Model
class Net(nn.Module):
    def __init__(self, InputSize, NumClasses):
        super(Net, self).__init__()
        # Linear layer 1: maps from input features to hidden layer
# It takes the input features (the sensor readings from bot) and maps them into a hidden layer of neurons.
        self.fc1 = nn.Linear(InputSize, HiddenSize)
        # Non-linear activation function introduces non-linearity
# Rectified Linear Unit introduces non-linearity by turning all negative outputs into zero while keeping positive ones unchanged
        self.relu = nn.ReLU()
        # Linear layer 2: maps from hidden layer to output
# Takes the transformed data from the hidden layer and maps it to the output space, where each output neuron corresponds to a possible decision
        self.fc2 = nn.Linear(HiddenSize, NumClasses)

    def forward(self, x):
        ###### Write Steps For Forward Pass Here! ######
        out = self.fc1(x)        # Step 1: input -> hidden
        out = self.relu(out)     # Step 2: apply non-linearity
        out = self.fc2(out)   
        return out

net = Net(InputSize, NumClasses)

###### Define The Loss Function Here! ######
criterion = nn.MSELoss() # Mean Squared Error Loss Function

optimizer = torch.optim.SGD(net.parameters(), lr = 1e-4) # Learning rate is set to 1e-4

criterion = nn.MSELoss() ###### Define The Loss Function Here! ######
learning_rate = [1e-6, 1e-4, 1e-2]
loss_records = {}
for lr in learning_rate:
    optimizer = torch.optim.SGD(net.parameters(), lr = lr) ###### Define The Optimizer Here! ######

##################################################################################################

if __name__ == "__main__":

    TrainSize, SensorNNData, SensorNNLabels = PreprocessData()
    for j in range(NumEpochs):
        losses = 0
        for i in range(TrainSize):
            input_values = Variable(SensorNNData[i])
            labels = Variable(SensorNNLabels[i])
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = net(input_values)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            losses += loss.item()

        print('Epoch %d, Loss: %.4f' % (j + 1, losses / SensorNNData.shape[0]))
        torch.save(net.state_dict(), './SavedNets/NNBot.pkl')

    plt.figure(figsize=(8, 6))
    for lr, losses in loss_records.items():
        plt.plot(range(1, NumEpochs + 1), losses, label=f'lr = {lr}', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Loss vs Epoch for Different Learning Rates', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

'''