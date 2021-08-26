import random
import numpy as np
import torch.nn as nn
import torch 
from torchvision import transforms, utils

class Pars(torch.nn.Module):

    def __init__(self, image, config, model, device):
        super(Pars, self).__init__()
        if image is None:
            if config is None:
                sideX, sideY = 256, 256
                batch_size = 1
            else:
                sideX, sideY = config.width, config.height
                batch_size = config.batch_size
            self.normu = .5*torch.randn(batch_size, 256, sideX//16, sideY//16).to(device)
            self.normu = torch.nn.Parameter(torch.sinh((0.5+2.5*random.random())*torch.arcsinh(self.normu)))
        else:
            self.image = np.array(image)#astype(np.float32)
            img = torch.unsqueeze(transforms.ToTensor()(self.image), 0) 
            img = 2.*img - 1.
            img = img.to(device).type(torch.cuda.FloatTensor)
            z, _, [_, _, indices] = model.encode(img)
            self.normu = torch.nn.Parameter(z.to(device))
    
    def forward(self):
        #normu = torch.nn.functional.softmax(self.normu, dim=-1)#.view(1, 8192, 64, 64)
        #return normu
        return self.normu.clip(-6, 6)
