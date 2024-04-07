import torch
import torch.nn as nn
import torch.nn.functional as F
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
            self.fc.append(nn.Linear(classifier_dims[i], classifier_dims[i+1]))
        self.fc.append(nn.Linear(classifier_dims[self.num_layers-1], classes))
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



class TMC(nn.Module):
    def __init__(self, classes, views, classifier_dims, lambda_epochs=1):
        """
        Main Trusted Multi-View Classification model, which integrates evidence from multiple views.
        
        classes: Number of classes 
        views: Number of views
        classifier_dims: classifier Dimension
        annealing_epoch: KL divergence annealing epoch during training
        """
        super(TMC, self).__init__()
        self.views = views
        self.classes = classes
        self.lambda_epochs = lambda_epochs
        self.Classifiers = nn.ModuleList([Classifier(classifier_dims[i], self.classes) for i in range(self.views)])

    def DS_Combin_two(self, alpha1, alpha2):
        """
        alpha1: Dirichlet distribution parameters of view 1
        alpha2: Dirichlet distribution parameters of view 2
        return: combined belief (b), evidence (E), and uncertainty (u) for each view.
        """
        alpha = dict()
        alpha[0], alpha[1] = alpha1, alpha2
        b, S, E, u = dict(), dict(), dict(), dict()
        for v in range(2):
            S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
            E[v] = alpha[v]-1
            b[v] = E[v]/(S[v].expand(E[v].shape))
            u[v] = self.classes/S[v]

        # b^0 @ b^(0+1)
        bb = torch.bmm(b[0].view(-1, self.classes, 1), b[1].view(-1, 1, self.classes))
        # b^0 * u^1
        uv1_expand = u[1].expand(b[0].shape)
        bu = torch.mul(b[0], uv1_expand)
        # b^1 * u^0
        uv_expand = u[0].expand(b[0].shape)
        ub = torch.mul(b[1], uv_expand)
        # calculate C
        bb_sum = torch.sum(bb, dim=(1, 2), out=None)
        bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
        C = bb_sum - bb_diag

        # calculate b^a
        b_a = (torch.mul(b[0], b[1]) + bu + ub)/((1-C).view(-1, 1).expand(b[0].shape))
        # calculate u^a
        u_a = torch.mul(u[0], u[1])/((1-C).view(-1, 1).expand(u[0].shape))

        # calculate new S
        S_a = self.classes / u_a
        # calculate new e_k
        e_a = torch.mul(b_a, S_a.expand(b_a.shape))
        alpha_a = e_a + 1
        return alpha_a

    def DS_Combin(self, alpha):
        """
        Aggregates Dirichlet parameters from all views (alpha) into a single set of parameters (alpha_a).
        Sequentially combines pairs of views using DS_Combin_two until all views are integrated.

        alpha:  Dirichlet distribution parameters.
        return: Combined Dirichlet distribution parameters.
        """
        for v in range(len(alpha)-1):
            if v==0:
                alpha_a = self.DS_Combin_two(alpha[0], alpha[1])
            else:
                alpha_a = self.DS_Combin_two(alpha_a, alpha[v+1])
        return alpha_a

    def forward(self, X, y, global_step):
        """
        The main method that processes multi-view input data (X), computes the evidence from each view, and aggregates it.
        """
        
        evidence = self.infer(X)
        loss = 0
        alpha = dict()
        for v_num in range(len(X)):
            alpha[v_num] = evidence[v_num] + 1
            loss += loss_functions.ce_loss(y, alpha[v_num], self.classes, global_step, self.lambda_epochs)
        alpha_a = self.DS_Combin(alpha)
        evidence_a = alpha_a - 1

        # Calculates the loss for each view using the custom ce_loss function from loss_functions
        # then combines the evidence and recalculates the loss for the aggregated evidence.
        
        loss += loss_functions.ce_loss(y, alpha_a, self.classes, global_step, self.lambda_epochs)
        loss = torch.mean(loss)
        return evidence, evidence_a, loss

    def infer(self, input):
        """
        Generates evidence for each view by passing the view-specific input through its corresponding Classifier.

        input: Multi-view data
        return: evidence for each view
        """
        evidence = dict()
        for v_num in range(self.views):
            evidence[v_num] = self.Classifiers[v_num](input[v_num])
        return evidence