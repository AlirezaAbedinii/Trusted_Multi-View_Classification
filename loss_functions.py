import torch
import torch.nn.functional as functions


def KL(alpha, c):
    """
    computes the Kullback-Leibler divergence between two Dirichlet distributions.
    beta: a tensor of ones that represents the parameters of the base Dirichlet distribution.
    alpha: sum of the alpha parameters for each observation (row in the alpha tensor).
    S_beta: corresponding sum for the beta tensor.
    lnB: log of the multivariate Beta function for the alpha parameters.
    lnB_uni: log of the multivariate Beta function for the beta parameters (the uniform distribution).
    dg0 and dg1: digamma function for S_alpha and alpha
    """

    beta = torch.ones((1, c)).cuda()
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def ce_loss(p, alpha, c, global_step, annealing_step):
    """
    Overall loss for the TMC
    S: sum of the alpha parameters (the strength of the Dirichlet distribution
    E: expected value of the Dirichlet distribution
    label: one-hot encodes the integer labels p into binary class matrix representations for multi-class classification.
    """
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = functions.one_hot(p, num_classes=c)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    annealing_coef = min(1, global_step / annealing_step)

    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c)

    return (A + B)
