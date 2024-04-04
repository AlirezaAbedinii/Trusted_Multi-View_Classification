import torch
import torch.nn as nn
import loss_functions
import warnings
warnings.filterwarnings("ignore")

class Classifier(nn.Module):
    """
    Defines a simple feedforward neural network (classifier) with fully connected layers,
    """
    def __init__(self, classifier_dims, classes):
        super(Classifier, self).__init__()
        #  list of dimensions for each layer in the neural network.
        self.num_layers = len(classifier_dims)

        self.fc = nn.ModuleList()
        for i in range(self.num_layers-1):
            self.fc.append(nn.Linear(classifier_dims[i], classifier_dims[i]))
        self.fc.append(nn.Linear(classifier_dims[self.num_layers], classes))
        self.fc.append(nn.Softplus())

    def forward(self, x):
        """
        Defines the forward pass of the classifier.
        Data (x) is passed through each layer sequentially.
        """
        h = self.fc[0](x)
        for i in range(1, len(self.fc)):
            h = self.fc[i](h)
        return h
