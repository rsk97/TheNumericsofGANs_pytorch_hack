import torch


class ConsensusOptimizer(torch.optim.Optimizer):
      
    # Init Method:
    def __init__(self, params, lr, alpha=0.1, beta=0.9, eps=1e-8, momentum=0.9):
        super(ConsensusOptimizer, self).__init__(params, defaults={'lr': lr})
        self.momentum = momentum
      
    # Step Method
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                p.data = p.data - group['lr']*p.grad.data**2

                # if p not in self.state:
                #     self.state[p] = dict(mom=torch.zeros_like(p.data))
                # mom = self.state[p]['mom']
                # mom = self.momentum * mom - group['lr'] * p.grad.data
                # p.data += mom