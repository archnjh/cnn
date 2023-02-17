import torch.nn as nn

class FcNet3(nn.Module):
    def __init__(self, **kwargs):
        super(FcNet3, self).__init__(**kwargs)
        
        self.hidden1 = nn.Linear(784, 800)
        self.active1 = nn.ReLU()
        self.output = nn.Linear(800, 10)


    def forward(self, img):
        img = img.view(-1, 784); 
        hid_out1 = self.hidden1(img)
        act_out1 = self.active1(hid_out1)
        output = self.output(act_out1)
        return output
