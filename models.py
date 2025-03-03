import torch
import torch.nn as nn
from torchvision.models import vgg19

conv={
    'conv1_1': 0,
    'conv2_1': 5,
    'conv3_1': 10,
    'conv4_1': 19,
    'conv5_1': 28,
    'conv4_2': 21 #content

}

class StyleTransfer(nn.Module):
    def __init__(self,):
        super(StyleTransfer, self).__init__()
        #TODO VGG load
        self.vgg19_model= vgg19(pretrained=True)
        self.vgg19_features= self.vgg19_model.features

        #TODO conv layer 분리
        self.style_layer=[conv['conv1_1'],conv['conv2_1'], conv['conv3_1'], conv['conv4_1'], conv['conv5_1']]
        self.content_layer=[conv['conv4_2']] 

    def forward(self,x,mode):
        #TODO slicing 해서 사용하기
        #x=vgg_model(x)이거는 입력값을 네트워크에 통과시킨 것 
        features=[]
        if mode=="style":
            for i in range(len(self.vgg19_features)):
                x=self.vgg19_features[i](x) 
                if i in self.style_layer:
                    features.append(x)
        elif mode=="content":
            for i in range(len(self.vgg19_features)):
                x=self.vgg19_features[i](x)
                if i in self.content_layer:
                    features.append(x)
        return features 