import torch
import torch.nn as nn

gamma = torch.ones((3, 3))
beta = torch.ones(3)

x = torch.ones((2, 3, 4, 5))
n, c, h, w = list(x.size())
# input is formatted as NCHW, here we need it to be NHWC
tx = x.permute(0, 2, 3, 1).contiguous()
tx = tx.view(-1, c)
tx2 = tx * tx
denominator = tx2.mm(gamma) + beta

print('gamma:', gamma.shape)
print('beta:', beta.shape)
print('x:', x.shape)
print('tx:', tx.shape)
print('denominator:', denominator.shape)
