import torch
import torch.nn as nn
import torch.nn.functional as F

'''Feed Forward Neural Network For Regression'''

############################################# FOR STUDENTS #####################################
class FullyConnectedNetwork(nn.Module):
    def __init__(self, input_dim, num_hidden_neurons, dropout_rte):
        super(FullyConnectedNetwork, self).__init__()

        ###### Define 5 Linear Layers ######
        self.h_0 = nn.Linear(input_dim, num_hidden_neurons) # Input dimension layer 
        self.h_1 = nn.Linear(num_hidden_neurons, num_hidden_neurons)
        self.h_2 = nn.Linear(num_hidden_neurons, num_hidden_neurons)
        self.h_3 = nn.Linear(num_hidden_neurons, num_hidden_neurons)
        self.h_4 = nn.Linear(num_hidden_neurons, 4)  # Output: [sinθ0, cosθ0, sinθ1, cosθ1]

        # Define Dropout Layer 
        self.drop = nn.Dropout(p=dropout_rte)

    def forward(self, x):
        # Forward Pass Using tanh Activations and Dropout 
        out = torch.tanh(self.h_0(x))
        out = self.drop(out)

        out = torch.tanh(self.h_1(out))
        out = self.drop(out)

        out = torch.tanh(self.h_2(out))
        out = self.drop(out)

        out = torch.tanh(self.h_3(out))
        out = self.drop(out)

        # Final layer : use tanh since outputs are sine/cosine values
        out = torch.tanh(self.h_4(out))

        return out
#################################################################################################
