import torch.nn as nn
import torch as th
import torch.nn.functional as F
from torch import optim

class AdaHessian(optim.Optimizer):
    def __init__(self,params,wd = 0, mc_iters=1, lr=.001, betas=(.9,.999),eps = 1e-8, control_variate=False):
        super(AdaHessian, self).__init__(params, defaults={'lr':lr})
        self.state = dict()
        self.lr = lr
        self.betas = betas
        self.control_variate = control_variate
        self.eps = eps
        self.mc_iters = mc_iters
        self.n_steps = 0
        self.wd = wd
        for group in self.param_groups:
            for p in group['params']:
                self.state[p] = dict(mom=th.zeros_like(p.data),hess_mom=th.zeros_like(p.data))
                p.hess = th.zeros_like(p.data)
    def zero_hessian(self):
        for group in self.param_groups:
            for p in group['params']:
                p.hess = th.zeros_like(p.data)

    def set_hessian(self):
        vals = []
        params = []
        if self.control_variate:
            for group in self.param_groups:
                for p in group['params']:
                    params.append(p)
                    vals.append(p.grad - self.state[p]["hess_mom"].detach() * p * self.betas[1])
        else:
            for group in self.param_groups:
                for p in group['params']:
                    params.append(p)
                    vals.append(p.grad)


        for iter in range(self.mc_iters):
          with th.no_grad():
            z_values = [th.randn_like(p.data) for p in params]
            hz_values = th.autograd.grad(vals, params,z_values,retain_graph = (iter != self.mc_iters - 1))
            for p, z, hz in zip(params, z_values, hz_values):
                p.hess += hz * z / self.mc_iters

    def step(self,closure = None):
        loss = None
        if closure is not None:
          loss = closure()
        self.n_steps += 1
        self.zero_hessian()
        self.set_hessian()
        beta0 = self.betas[0]
        beta1 = self.betas[1]
        bias_correction_0 = 1-beta0**self.n_steps
        bias_correction_1 = 1-beta1**self.n_steps
        with th.no_grad():
          for group in self.param_groups:
              step_size = group['lr']*bias_correction_0
              for p in group['params']:
                  if self.wd != 0:
                      p.data = p.data * self.wd

                  mom = self.state[p]['mom']
                  mom.mul_(beta0).add_(p.grad,alpha=1-beta0)
                  hess_mom = self.state[p]['hess_mom']
                  hess_mom.mul_(beta1).add_(p.hess,alpha=1-beta1)
                  denominator = th.abs(hess_mom/bias_correction_1).pow(1/2).add_(self.eps)
                  p.addcdiv_(mom,denominator, value=-step_size)

        return loss



