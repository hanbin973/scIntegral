import torch
import torch.optim as optim
import torch.distributions as distributions

import numpy as np
import pandas as pd

# negative binomial likelihood
def nb_logpmf(k, n, p):
	log_normalized = torch.lgamma(n+k) - torch.lgamma(1+k) - torch.lgamma(n)
	log_unnormalized_prob = k * torch.log(p) + n*torch.log(1-p)

	return log_normalized + log_unnormalized_prob

# restrict phi range from 1~100
def phi_mod(phi):
	return phi.clamp(1, 100)

def phi_mod_grad(phi):
	return ((phi-1)*(phi-100)<0).float()

# gradient 
class custom_LL(torch.autograd.Function):
	
	@staticmethod
	def forward(ctx, Delta, Beta, Phi, Y, X, S, mask):
		with torch.no_grad():
			# modify Phi
			Phi_mod = phi_mod(Phi)

			# n x g x t
			Mu = S[:,None,None] * (X.mm(Beta)[:,:,None] + (mask*Delta)[None,:,:]).exp()
			
			# compute some values that are used repeatedly
			probs = Phi_mod[None,:,None]/(Mu + Phi_mod[None,:,None])
			ypr = Y[:,:,None] + Phi_mod[None,:,None]

			# n x g x t -> n x t
			logP = nb_logpmf(Y[:,:,None], Phi_mod[None,:,None], 1-probs).sum(dim=1)

			# n x t -> n
			logP_reduced = logP.logsumexp(axis=1)

			# n -> 1
			LL = logP_reduced.sum()

			# gamma : n x t
			Gamma = (logP - logP_reduced[:,None]).exp()

			# gradient
			A = (Mu - Y[:,:,None]) * probs
			D = torch.digamma(ypr) - torch.digamma(Phi_mod)[None,:,None] + \
					torch.log(probs) + A/(Mu + Phi_mod[None,:,None])
			dmoddpi = phi_mod_grad(Phi)

			grad_Delta = (A * Gamma[:,None,:]).mean(axis=0) * mask
			grad_Beta = ((A @ Gamma[:,:,None]) @ X[:,None,:]).mean(axis=0).T
			grad_Phi = dmoddpi * (-D * Gamma[:,None,:]).sum(axis=2).sum(axis=0)

			# save gradient to ctx
			ctx.save_for_backward(grad_Delta, grad_Beta, grad_Phi)

		return LL

	@staticmethod
	def backward(ctx, grad_output):
		grad_Y = grad_X = grad_S = grad_mask = None
		grad_Delta, grad_Beta, grad_Phi = ctx.saved_tensors
		
		return grad_Delta, grad_Beta, grad_Phi, grad_Y, grad_X, grad_S, grad_mask

def get_LL(Delta, Beta, Phi, Y, X, S, mask):
	with torch.no_grad():
		# modify phi
		Phi_mod = phi_mod(Phi)
		# n x g x t
		Mu = S[:,None,None] * (X.mm(Beta)[:,:,None] + (mask*Delta)[None,:,:]).exp()
		# repeatedly used values
		probs = Phi_mod[None,:,None]/(Mu + Phi_mod[None,:,None])
		# n x g x t -> n x t
		logP = nb_logpmf(Y[:,:,None], Phi_mod[None,:,None], 1-probs).sum(dim=1)
		# n x t -> n
		logP_reduced = logP.logsumexp(axis=1)
		# n -> 1
		LL = logP_reduced.sum()

	return LL

def get_Gamma(Delta, Beta, Phi, Y, X, S, mask):
	with torch.no_grad():
		# modify phi
		Phi_mod = phi_mod(Phi)
		# n x g x t
		Mu = S[:,None,None] * (X.mm(Beta)[:,:,None] + (mask*Delta)[None,:,:]).exp()
		# repeatedly used values
		probs = Phi_mod[None,:,None]/(Mu + Phi_mod[None,:,None])
		# n x g x t -> n x t
		logP = nb_logpmf(Y[:,:,None], Phi_mod[None,:,None], 1-probs).sum(dim=1)
		# n x t -> n
		logP_reduced = logP.logsumexp(axis=1)

	return (logP - logP_reduced[:,None]).exp()

def inference(Y, X, S, t, mask,
		n_itr_max=1000,
		err_bound=1e-4,
		use_gpu=False):
	
	"""
	Y : N (# of samples) x G (# of genes) matrix : expression information
	X : N (# of samples) x P (# of covariates) matrix : covariate information
	S : N (# of samples) vector : size factor
	mask : G (# of genes) x T (# of types) : marker information
	
	n_itr_max : maximum # of iteration of optimization, default : 1000
	err_bound : bound of relative error of log-likelihood, default : 1e-4
	use_gpu : True if using gpu and False if not, default : False

	"""
	# check input size
	if Y.shape[0] != S.shape[0]:
		raise("row # of Y and length of S do not match!")
	
	if Y.shape[0] != X.shape[0]:
		raise("row # of Y and row # of X do not match")

	if t != mask.shape[1]:
		raise("# of types and col # of mask do not match")

	if Y.shape[1] != mask.shape[0]:
		raise("col # of Y and row # of mask do not match")


	# initialize parameters
	n, g, p, t = Y.shape[0], Y.shape[1], X.shape[1], mask.shape[1]

	if use_gpu == True:
		# beta
		beta_g0 = Y.mean(dim=0).log()[None,:] * 0.8
		beta_gp = torch.zeros(size=(p-1,g)).cuda()
		beta = torch.cat((beta_g0, beta_gp))
		Beta = beta.detach().requires_grad_(True)

		# delta
		Delta = (mask * distributions.normal.Normal(loc=2, scale=0.05).sample((g,t)).cuda().clamp(2)).detach().requires_grad_(True)

		# phi
		Phi = (2 * torch.ones(g)).cuda().detach().requires_grad_(True)
	
	else:
		# beta
		beta_g0 = (Y.mean(dim=0)/t).log()[None,:]
		beta_gp = torch.zeros(size=(p-1,g))
		beta = torch.cat((beta_g0, beta_gp))
		Beta = beta.detach().requires_grad_(True)

		# delta
		Delta = (mask * distributions.normal.Normal(loc=2, scale=0.05).sample((g,t)).clamp(2)).detach().requires_grad_(True)

		# phi
		Phi = (2 * torch.ones(g)).cuda().detach().requires_grad_(True)
	
	# compute loss
	loss_old = -get_LL(Delta, Beta, Phi, Y, X, S, mask)
	print('Loss at initialization: %.3f' % loss_old)

	# define optimizer
	optimizer = optim.Adam([Delta, Beta, Phi], lr=0.05)

	def closure():
		optimizer.zero_grad()
		loss = -custom_LL.apply(Delta, Beta, Phi, Y, X, S, mask)
		loss.backward()
		return loss

	# start optimization
	print('Start optimization')
	for itr in range(n_itr_max):

		# one-step LBFGS
		optimizer.step(closure)

		# compute loss
		loss_new = -get_LL(Delta, Beta, Phi, Y, X, S, mask)
		if itr % 20 == 0:
			print('%d-th iteration, Loss: %.3f' % (itr, loss_new))

		# halt if convergence criteria is met
		if (loss_old - loss_new)/loss_old < err_bound:
			break

		# update loss
		loss_old = loss_new

	return Delta, Beta, Phi

def get_cell_types(Delta, Beta, Phi, Y, X, S, mask):

	# get parameters
	Gamma = get_Gamma(Delta, Beta, Phi, Y, X, S, mask)

	# compute type : length n-vector as numpy array
	cell_types = Gamma.argmax(axis=1).detach().cpu().numpy()

	return cell_types

def classify(Y, X, S, t, mask,
		n_itr_max=1000,
		err_bound=1e-4,
		use_gpu=False):

	if type(mask) == pd.core.frame.DataFrame:
		mask_torch = torch.tensor(mask.values, dtype=torch.float32)
		cell_type_list = mask.columns
	else:
		mask_torch = mask
	
	# get parameters
	Delta, Beta, Phi = inference(Y, X, S, t, mask_torch, n_itr_max, err_bound, use_gpu)
	Gamma = get_Gamma(Delta, Beta, Phi, Y, X, S, mask_torch)

	# compute type : length n-vector as numpy array
	cell_types = get_cell_types(Delta, Beta, Phi, Y, X, S, mask_torch)

	if type(mask) == pd.core.frame.DataFrame:
		cell_types = cell_type_list[cell_types]
		post_probs = pd.DataFrame(Gamma.detach().cpu().numpy(), columns=cell_type_list)
	else:
		post_probs = Gamma.detach().cpu().numpy()


	return cell_types, post_probs
