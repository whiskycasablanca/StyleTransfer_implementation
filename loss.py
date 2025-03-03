#content loss
#vgg19 feature map

#style losee
#gram matrix

import torch
import torch.nn as nn
import torch.nn.functional as F #MSE 이용하기 위해 필요함

class ContentLoss(nn.Module):
    def __init__(self,):
        super(ContentLoss, self).__init__()

    def forward(self, x:torch.Tensor, y:torch.Tensor):
        # MSE Loss
        loss=F.mse_loss(x,y)
        return loss
    

class StyleLoss(nn.Module):
    def __init__(self,):
        super(StyleLoss, self).__init__()
    
    def gram_matrix(self, x:torch.Tensor):
        b,c,h,w=x.size()
        #reshape
        features= x.view(b,c,h*w)
        features_T=features.transpose(1,2)
        G=torch.matmul(features,features_T)

        return G.div(b*c*h*w)
    
    def forward(self, x, y):
        Gx=self.gram_matrix(x)
        Gy=self.gram_matrix(y)
        loss= F.mse_loss(Gx,Gy)
        return loss