import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch.nn import init
from Gdn import Gdn2d, Gdn1d
from Spp import SpatialPyramidPooling2d

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight.data)
    elif classname.find('Gdn2d') != -1:
        init.eye_(m.gamma.data)
        init.constant_(m.beta.data, 1e-4)
    elif classname.find('Gdn1d') != -1:
        init.eye_(m.gamma.data)
        init.constant_(m.beta.data, 1e-4)

class CheckNan(nn.Module):
    def forward(self, x):
        if x.isnan().any():
            print('>>>>>>> nan in x')
        return x

def build_model(normc=Gdn2d, normf=Gdn1d, layer=3, width=48, outdim=2):
    layers = [
        CheckNan(),
        nn.Conv2d(3, width, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
        CheckNan(),
        normc(width),
        nn.MaxPool2d(kernel_size=2)
    ]

    for l in range(1, layer):
        layers += [nn.Conv2d(width,  width, kernel_size=3, stride=1, padding=1,  dilation=1,  bias=True),
                    CheckNan(),
                   normc(width),
                    CheckNan(),
                   nn.MaxPool2d(kernel_size=2)
                   ]

    layers += [nn.Conv2d(width,  width, kernel_size=3, stride=1, padding=1,  dilation=1,  bias=True),
               normc(width),
               SpatialPyramidPooling2d(pool_type='max_pool')
               ]
    layers += [nn.Linear(width*14, 128, bias=True),
               nn.ReLU(),
               nn.Linear(128, outdim, bias=True)
               ]
    net = nn.Sequential(*layers)
    net.apply(weights_init)

    return net


class E2EUIQA(nn.Module):
    # end-to-end unsupervised image quality assessment model
    def __init__(self, config):
        super(E2EUIQA, self).__init__()
        if config.std_modeling and not config.fixvar:
            outdim = 2
        else:
            outdim = 1
        self.cnn = build_model(outdim=outdim)
        self.config = config

    def forward(self, x):
        r = self.cnn(x)
        #!print('Shape of r:', r.shape)    #<
        mean = r[:, 0]
        if self.config.fixvar:
            # var = torch.tensor(1.0, dtype=mean.dtype, device=mean.device)
            var = torch.ones_like(mean)
        else:
            var = torch.exp(r[:, 1])

        #!print('Shape of mean:', mean.shape)    #<
        #!input()           #<

        #!print('The shape of r:', r.shape)
        #< 128, 2
        #!print('Shape of mean:', mean.shape)    #<
        return mean, var

    def init_model(self, path):
        self.cnn.load_state_dict(torch.load(path))

    def gdn_param_proc(self):
        for m in self.modules():
            #!!print('------>>>')
            if isinstance(m, Gdn2d) or isinstance(m, Gdn1d):
                m.beta.data.clamp_(min=2e-10)
                m.gamma.data.clamp_(min=2e-10)
                m.gamma.data = (m.gamma.data + m.gamma.data.t()) / 2
