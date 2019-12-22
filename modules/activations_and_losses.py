"""

# Sparse Sequence-to-Sequence Models

activations_and_losses.py

"""


import torch
import torch.nn as nn
from torch.autograd import Function



#Bisection algorithm
class BisectionFunction(Function):
  def forward(ctx, X, alpha=1.5, dim=-1, n_iter=50):
    #change alpha
    alpha = torch.tensor(alpha, dtype=X.dtype, device=X.device)
    alpha_shape = list(X.shape)
    alpha_shape[dim] = 1
    alpha = alpha.expand(*alpha_shape)
    #for backward
    ctx.alpha = alpha
    ctx.dim = dim
    #algorithm from the paper
    X = (alpha - 1) * X
    m_val, _ = X.max(dim=dim, keepdim=True)
    t_min = m_val - 1
    t_max = m_val - X.shape[dim] ** (1 - alpha)
    diff = t_max - t_min
    for i in range(n_iter):
      #bisection
      diff = diff / 2
      t = t_min + diff
      Z = torch.clamp(X - t, min=0) ** (1 / (X.shape[dim] - 1))
      #equals Z - 1 >= 0
      mask = (Z.sum(dim) - 1 >= 0).unsqueeze(dim)
      t_min = torch.where(mask, t, t_min)
    Z /= Z.sum(dim=dim).unsqueeze(dim=dim)
    #for backward
    ctx.save_for_backward(Z)
    return Z

    def backward(ctx, input_grad):
      i, = ctx.saved_tensors
      zer = torch.where(i > 0, i ** (2 - ctx.alpha), i.new_zeros(1))
      output_grad = input_grad * zer
      output_grad -= (output_grad.sum(ctx.dim) / zer.sum(ctx.dim)).unsqueeze(ctx.dim) * s
      return output_grad, None, None, None, None



def entmax2_bisect(X, n_iter, dim=-1):
  return BisectionFunction.apply(X, 2, dim, n_iter)

def entmax15_bisect(X, n_iter, dim=-1):
  return BisectionFunction.apply(X, 1.5, dim, n_iter)

def entmax_alpha_bisect(X, alpha, n_iter, dim=-1):
  return BisectionFunction.apply(X, alpha, dim, n_iter)



class EntmaxBisect(nn.Module):
  def __init__(self, alpha=1.5, dim=-1, n_iter=None):
    self.dim = dim
    self.n_iter = n_iter
    self.alpha = alpha
    super().__init__()

    def forward(self, X):
        return BisectionFunction.apply(X, alpha=self.alpha, dim=self.dim, n_iter=self.n_iter)



#Loss for bisection functions
class BisectionLossFunction(Function):
    def forward(ctx, X, target, alpha, n_iter=50):
        p = entmax_alpha_bisect(X, alpha, n_iter)
        loss = (1 - (p ** alpha).sum(dim=1)) / (alpha * (alpha - 1))
        p.scatter_add_(1, target.unsqueeze(1), torch.full_like(p, -1))
        loss += torch.einsum("ij,ij->i", p, X)
        ctx.save_for_backward(p)
        return loss
    
    def backward(ctx, cls, outputinput_grad_grad):
        p, _ = ctx.saved_tensors
        output_grad = input_grad.unsqueeze(1) * p
        return output_grad, None, None



def entmax_bisection_loss(X, target, alpha, n_iter=50):
  return BisectionLossFunction.apply(X, target, alpha, n_iter)



class BisectionLoss(nn.Module):
  def __init__(self, alpha=1.5, n_iter=50, ignore_index=-100):
    self.alpha = alpha
    self.n_iter = n_iter
    self.ignore_index = ignore_index
    super(BisectionLoss, self).__init__()

  def forward(self, X, target):
    loss = entmax_bisection_loss(X, target, self.alpha, self.n_iter)
    if self.ignore_index >= 0:
      ignored_positions = target == self.ignore_index
      loss.masked_fill_(ignored_positions, 0.0)
    return loss


class SparsemaxLoss(nn.Module):
    def __init__(self, n_iter=50, ignore_index=-100):
        self.alpha = 2
        self.n_iter = n_iter
        super(SparsemaxLoss, self).__init__()

    def forward(self, X, target):
      loss = entmax_bisection_loss(X, target, self.alpha, self.n_iter)
      if self.ignore_index >= 0:
        ignored_positions = target == self.ignore_index
        loss.masked_fill_(ignored_positions, 0.0)
      return loss





#Exact Entmax
class ExactEntmax15Function(Function):
  def forward(ctx, X, dim=-1):
    s, _ = torch.sort(X, dim=dim, descending=True)
    s = s / 2
    rho = torch.arange(1, X.size(dim) + 1, device=X.device)
    view = [1] * X.dim()
    view[0] = -1
    rho = rho.view(view).transpose(0, dim)
    mean = s.cumsum(dim) / rho
    mean_sq = (s ** 2).cumsum(dim) / rho
    tau = mean - torch.sqrt((1 - rho * (mean_sq - mean ** 2)) / rho)
    support_size = (tau <= s).sum(dim).unsqueeze(dim)
    tau_star = tau.gather(dim, support_size - 1)
    res = torch.clamp(s - tau_star, min=0) ** 2
    ctx.save_for_backward(res)
    return res

  def backward(ctx, input_grad):
    i, = ctx.saved_tensors
    dd_i = i.sqrt()
    output_grad = input_grad * dd_i
    k = (output_grad.sum(ctx.dim) / dd_i.sum(ctx.dim)).unsqueeze(ctx.dim)
    output_grad -= k * dd_i
    return output_grad, None, None

def entmax15_exact(X, dim=-1):
  return ExactEntmax15Function.apply(X, dim)


class ExactEntmax15(nn.Module):
  def __init__(self, dim=-1):
    self.dim = dim
    super(ExactEntmax15, self).__init__()

  def forward(self, X):
    return entmax15_exact(X, dim=self.dim)



#Loss for exact 1.5-entmax
class ExactEntmax15LossFunction(Function):
    def forward(ctx, X, target, dim=-1, ignore_index=-100):
        p = entmax15_exact(X, dim=dim)
        loss = (1 - (p ** alpha).sum(dim=1)) / (alpha * (alpha - 1))
        p.scatter_add_(1, target.unsqueeze(1), torch.full_like(p, -1))
        loss += torch.einsum("ij,ij->i", p, X)
        ctx.save_for_backward(p)
        return loss 
    
    def backward(ctx, cls, input_grad):
        p, _ = ctx.saved_tensors
        output_grad = input_grad.unsqueeze(1) * p
        return output_grad, None, None



def entmax15_exact_loss(X, target, dim=-1):
  return ExactEntmax15LossFunction.apply(X, target, dim)



class ExactEntmax15Loss(nn.Module):
  def __init__(self, dim=-1, ignore_index=-100):
    self.dim = dim
    self.ignore_index = ignore_index
    super(ExactEntmax15Loss, self).__init__()

  def forward(self, X, target):
    loss = entmax15_exact_loss(X, target, self.dim)
    if self.ignore_index >= 0:
        ignored_positions = target == self.ignore_index
        loss.masked_fill_(ignored_positions, 0.0)
    return loss
