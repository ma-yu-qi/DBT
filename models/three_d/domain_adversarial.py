# models/domain_adversarial.py
import torch
import torch.nn as nn
from torch.autograd import Function

# 定义梯度反转层
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

def grad_reverse(x, lambda_=1.0):
    return GradientReversalFunction.apply(x, lambda_)

class DomainClassifier(nn.Module):
    def __init__(self, feature_dim, num_domains):
        super(DomainClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, num_domains)
        )
    
    def forward(self, x, lambda_=1.0):
        reversed_x = grad_reverse(x, lambda_)
        out = self.classifier(reversed_x)
        return out
