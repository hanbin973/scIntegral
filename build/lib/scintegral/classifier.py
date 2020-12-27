import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Function

import numpy as np
import pandas as pd

def nb_logpmf(k, n, p):
    log_norm_const = torch.lgamma(n+k) - torch.lgamma(1+k) - torch.lgamma(n)
    log_unnormalized_prob = k * torch.log(p) + n * torch.log(1-p)
    return log_norm_const + log_unnormalized_prob

def phi_filter(phi):
    return phi.clamp(1, 100)

def phi_filter_grad(phi):
    return ((phi-1)*(phi-100) < 0).float()

def posterior_probs(delta, beta, phi, expr_mat, cov_mat, size_factor, marker_onehot):
    with torch.no_grad():
        # modify phi
        phi_filtered = phi_filter(phi)

        # n x g x t
        mu = size_factor[:,None,None] * \
            (cov_mat.mm(beta)[:,:,None] + (marker_onehot * delta)[None,:,:]).exp()

        # some preliminary calculations
        probs = phi_filtered[None,:,None]/(mu + phi_filtered[None,:,None])

        # n x t
        log_probs = nb_logpmf(expr_mat[:,:,None], phi_filtered[None,:,None], 1-probs).sum(dim=1)

        # n x t -> n
        log_prob = log_probs.logsumexp(axis=1)

        # n -> out
        out = log_prob.sum()

        # compute posterior prob
        gamma = (log_probs - log_prob[:,None]).exp()

    return gamma.argmax(dim=1)

# ref: https://pytorch.org/docs/stable/notes/extending.html
class scintegral_loss(Function):
    @staticmethod
    def forward(ctx, delta, beta, phi, expr_mat, cov_mat, size_factor, marker_onehot):
        with torch.no_grad():
            # modify phi
            phi_filtered = phi_filter(phi)

            # n x g x t
            mu = size_factor[:,None,None] * \
                (cov_mat.mm(beta)[:,:,None] + (marker_onehot * delta)[None,:,:]).exp()

            # some preliminary calculations
            probs = phi_filtered[None,:,None]/(mu + phi_filtered[None,:,None])
            ypr = expr_mat[:,:,None] + phi_filtered[None,:,None]

            # n x t
            log_probs = nb_logpmf(expr_mat[:,:,None], phi_filtered[None,:,None], 1-probs).sum(dim=1)

            # n x t -> n
            log_prob = log_probs.logsumexp(axis=1)

            # n -> out
            out = log_prob.sum()

            # -- start computing gradient --
            gamma = (log_probs - log_prob[:,None]).exp()
            A = (mu - expr_mat[:,:,None]) * probs
            D = torch.digamma(ypr) - torch.digamma(phi_filtered)[None,:,None] + \
                torch.log(probs) + A/(mu + phi_filtered[None,:,None])
            d_phi_filter = phi_filter_grad(phi)

            grad_delta = (A * gamma[:,None,:]).sum(axis=0) * marker_onehot
            grad_beta = (A @ gamma[:,:,None] @ cov_mat[:,None,:]).sum(axis=0).T
            grad_phi = d_phi_filter * (-D * gamma[:,None,:]).sum(axis=2).sum(axis=0)

            # save for backward
            ctx.save_for_backward(grad_delta, grad_beta, grad_phi)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_delta, grad_beta, grad_phi = ctx.saved_tensors

        return grad_delta, grad_beta, grad_phi, None, None, None, None

class scintegral_loss_nograd(Function):
    @staticmethod
    def forward(ctx, delta, beta, phi, expr_mat, cov_mat, size_factor, marker_onehot):
        with torch.no_grad():
            # modify phi
            phi_filtered = phi_filter(phi)

            # n x g x t
            mu = size_factor[:,None,None] * \
                (cov_mat.mm(beta)[:,:,None] + (marker_onehot * delta)[None,:,:]).exp()

            # some preliminary calculations
            probs = phi_filtered[None,:,None]/(mu + phi_filtered[None,:,None])

            # n x t
            log_probs = nb_logpmf(expr_mat[:,:,None], phi_filtered[None,:,None], 1-probs).sum(dim=1)

            # n x t -> n
            log_prob = log_probs.logsumexp(axis=1)

            # n -> out
            out = log_prob.sum()

        return out

class scintegral_model(nn.Module):
    def __init__(self, expr_mat, cov_mat, size_factor, n_type, marker_onehot,
                prior_mean,
                prior_width,
                disp_init):
        super().__init__()

        # save size parameters
        self.n_sample = expr_mat.shape[0]
        self.n_gene = expr_mat.shape[1]
        self.n_batch = cov_mat.shape[1]
        self.n_type = n_type

        # save data
        self.expr_mat = expr_mat
        self.cov_mat = cov_mat
        self.size_factor = size_factor
        self.marker_onehot = marker_onehot

        # generate initial parameters
        self.delta = nn.Parameter((prior_width * torch.randn((self.n_gene, self.n_type)) + prior_mean).clamp(prior_mean) * self.marker_onehot.to(torch.device('cpu')))
        self.phi = nn.Parameter(disp_init * torch.ones(self.n_gene))

        # initializing beta needs some work
        A = cov_mat.T @ cov_mat
        B = cov_mat.T @ (expr_mat / size_factor[:,None])
        alpha, _ = torch.solve(B, A)
        beta_intercept = alpha[0,:].clamp(1e-2,1e3).log()
        beta_coef = (1 + alpha[1:,:]/alpha[0,:][None,:]).clamp(1e-2,1e3).log()
        self.beta = nn.Parameter(torch.cat((beta_intercept[None,:], beta_coef))*0.5)

    def forward(self, only_loss=False):
        if only_loss == False:
            out = scintegral_loss.apply(self.delta, self.beta, self.phi, self.expr_mat, self.cov_mat, self.size_factor, self.marker_onehot)
        elif only_loss == True:
            out = scintegral_loss_nograd.apply(self.delta, self.beta, self.phi, self.expr_mat, self.cov_mat, self.size_factor, self.marker_onehot)

        return out

def classify_cells_internal(expr_mat, cov_mat, size_factor, n_type, marker_onehot,
                    device,
                    e_converge,
                    lr,
                    n_itr_max,
                    prior_mean,
                    prior_width,
                    disp_init):

    # move data to device
    expr_mat = expr_mat.to(device)
    cov_mat = cov_mat.to(device)
    size_factor = size_factor.to(device)
    marker_onehot = marker_onehot.to(device)

    # initialize model
    model = scintegral_model(expr_mat, cov_mat, size_factor, n_type, marker_onehot, prior_mean, prior_width, disp_init)
    model.to(device)

    optim_all = optim.Adam(model.parameters(), lr=lr)
    def closure_all():
        optim_all.zero_grad()
        loss = -model(only_loss=False)
        loss.backward()
        return loss

    loss_old = -model(only_loss=True)
    for itr in range(n_itr_max):
        # optimize all parameters
        optim_all.step(closure_all)

        # get current loss and update
        loss_new = -model(only_loss=True)
        if (loss_old - loss_new) / torch.abs(loss_old) < e_converge :
            break
        loss_old = loss_new

    # infer cell types
    cell_types = posterior_probs(model.delta, model.beta, model.phi, expr_mat, cov_mat, size_factor, marker_onehot)

    return model, cell_types

"""
        Classifier
        ~~~~~~~~~~~


"""

def classify_cells(expr_mat, cov_mat, size_factor, marker_onehot,
                    device=torch.device('cpu'),
                    e_converge=1e-4,
                    lr=0.05,
                    n_itr_max=1000,
                    prior_mean=2,
                    prior_width=0.05,
                    disp_init=2):
    """
    The cell-type classifier.


    :param expr_mat: An n (number of cells) x g (number of genes) matrix that contains the raw expression counts.
    :param cov_mat: An n x p (number of batches) matrix that contains the batch membership of samples.
    :param size_factor: An n vector of size factors (or any offset variable replacing the size factor).
    :param marker_onehot: A g x t (number of cell-types) matrix containing marker information for each cell-type.
    :param device: A device used in computation (default: cpu).
    :param e_converge: A real number used to determine convergence (default:1e-4).
    :param lr: A real number used as the initial learning-rate (default:0.2).
    :param n_itr_max: The maximum number of iterations in the likielihood optimization step (default:1000).
    :param prior_mean: The threshold parameter for initialization (default:2).
    :param prior_with: The width parameter for initialization (default:0.05).
    :param disp_init: The initial dispersion parameter of the negative binomial likelihood (default:1.5).


    :returns list: A list of assigned cell-type labels.
    :returns class: A torch module object containing the fitted parameters.

    """

    # set dtype to float32
    dtype = torch.float32
    torch.set_default_dtype(dtype)

    # save type names
    type_names = marker_onehot.columns
    n_type = type_names.shape[0]

    # convert numpy arrays to torch tensors
    expr_mat = torch.tensor(expr_mat, dtype=dtype)
    cov_mat = torch.tensor(cov_mat, dtype=dtype)
    size_factor = torch.tensor(size_factor, dtype=dtype)
    marker_onehot = torch.tensor(marker_onehot.values, dtype=dtype)

    model, cell_types_int = classify_cells_internal(expr_mat, cov_mat, size_factor, n_type, marker_onehot,
                                                device=device,
                                                e_converge=e_converge,
                                                lr=lr,
                                                n_itr_max=n_itr_max,
                                                prior_mean=prior_mean,
                                                prior_width=prior_width,
                                                disp_init=disp_init)

    cell_types = [type_names[i] for i in cell_types_int.detach().cpu().numpy()]

    return cell_types, model
