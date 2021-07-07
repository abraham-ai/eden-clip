import torch.nn as nn
import torch 

class Pars(torch.nn.Module):

    def __init__(self, image, config, device):
        super(Pars, self).__init__()
        if image is None:
            if config is None:
                sideX, sideY, channels = 256, 256, 3
                batch_size = 1
            else:
                sideX, sideY, channels = config.size[0], config.size[1], 3
                batch_size = config.batch_size
            self.normu = .5*torch.randn(batch_size, 256, sideX//16, sideY//16).to(device)
            self.normu = torch.nn.Parameter(torch.sinh(1.9*torch.arcsinh(self.normu)))
        else:
            self.image = np.array(image)#astype(np.float32)
            img = torch.unsqueeze(transforms.ToTensor()(self.image), 0) 
            img = 2.*img - 1.
            img = img.to(DEVICE).type(torch.cuda.FloatTensor)
            z, _, [_, _, indices] = taming_transformers.model.encode(img)
            self.normu = torch.nn.Parameter(z.to(device))
    
    def forward(self):
        #normu = torch.nn.functional.softmax(self.normu, dim=-1)#.view(1, 8192, 64, 64)
        #return normu
        return self.normu.clip(-6, 6)
