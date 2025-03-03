#기본적인 것들 import하기
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import os

import numpy as np
from PIL import Image

from models import StyleTransfer
from loss import ContentLoss, StyleLoss
from tqdm import tqdm

#pre trained된 모델 사용할 때는 그 모델이 사용했던 pre processing방법을 그대로 따라해야된다. 
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def pre_processing(image:Image.Image)-> torch.Tensor:
    #이미지 resize (512,512)
    #Image -> Tensor
    #Normalize
    
    preprocessing=T.Compose([
        T.Resize((512,512)),
        T.ToTensor(),
        T.Normalize(mean,std)
    ]) #(c,h,w)
    
    #밑의 post processing 코드와 맞춰주기 위해서 앞에 배치차원 1 추가해주자
    #(1,c,h,w)
    image_tensor:torch.Tensor= preprocessing(image)
    return image_tensor.unsqueeze(0)



def post_processing(tensor:torch.Tensor)-> Image.Image:
    #https://chatgpt.com/share/67c414d1-474c-8005-b68c-574c0df04820
    #shape 1,c,h,w- 이미지 하나니까 배치1이다
    # PIL로 이미지 출력하려면 텐서를 넘파이 형식으로 수정해야 한다. 
    # 파이토치 텐서는 기본적으로 텐서.grad에 기울기 정보가 저장되어 있다. numpy는 관련 기능(미분계산 등)이 제공되지 않으므로 넘파이로 변환하려면 필요하다
    image:np.ndarray= tensor.to('cpu').detach().numpy()
    #shape c,h,w -앞에꺼 떼주기 squeeze로
    image=image.squeeze()
    #shape h,w,c- PIL은 h,w,c 형식이다. 
    image=image.transpose(1,2,0)
    #de normalization
    image=image*std+mean
    #클리핑 해주기(이상치처리)
    image=image.clip(0,1)*255
    #dtype uint8
    image=image.astype(np.uint8)
    #numpy->Image
    return Image.fromarray(image)

def train_main():
#이미지 불러오기(평소에는 데이터셋)
    content_image=Image.open('./content_disastergirl.jpg')
    content_image=pre_processing(content_image)

    #style_image=Image.open('./style.jpg')
    style_image=Image.open('./style_hockney.jpg')
    style_image=pre_processing(style_image)


##이미지 pre,post processing하기-텐서변환, 차원 맞추기


    #models, loss 로드하기
    style_transfer=StyleTransfer()

    content_loss=ContentLoss()
    style_loss=StyleLoss()
    
    #하이퍼파라미터
    alpha=1  #각 로스의 비율
    beta=1e6
    lr=0.1

    save_root= f'{alpha}_{beta}_{lr}'
    # save_root= f'{alpha}_{beta}_{lr}'
    os.makedirs(save_root, exist_ok=True)

    #디바이스 세팅
    device='cpu'
    if torch.cuda.is_available():
        device='cuda'

    style_transfer=style_transfer.to(device)
    content_image=content_image.to(device)
    syle_image=style_image.to(device)
    ## 노이즈 이미지에서 시작하는 경우
    ###x=torch.randn(1,3,512,512).to(device)
    ## 컨텐츠 이미지에서 시작하는 경우
    x=content_image.clone()
    x.requires_grad_(True)
    

    #옵티마이저 설정하기
    
    optimizer=optim.Adam([x],lr=lr)

    #train loop설정하기
    steps=1000
    for step in tqdm(range(steps)):
        ##content represenation (x, content_image)
        ##style representaion (x, style_image)

        #아웃풋이 피처맵의 리스트니까
        x_content_list= style_transfer(x,'content') 
        y_content_list= style_transfer(content_image,'content')

        x_style_list= style_transfer(x,'style') 
        y_style_list= style_transfer(style_image,'style')
        
        ##loss_content, loss_style
        loss_c=0
        loss_s=0
        loss_total=0

        #style의 features는 5개 content는 1개
        for x_content, y_content in zip(x_content_list, y_content_list):
            loss_c+=content_loss(x_content,y_content)
        loss_c=alpha*loss_c

        for x_style, y_style in zip(x_style_list, y_style_list):
            loss_c+=style_loss(x_style,y_style)
        loss_s=beta*loss_c

        loss_total=loss_c + loss_s
    
        ## optimizer step
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        if step%100==0:
            print(f"loss_c: {loss_c.cpu()}")
            print(f"loss_s: {loss_s.cpu()}")
            print(f"loss_total: {loss_total.cpu()}")

            gen_img:Image.Image= post_processing(x)
            #운영체제에 맞게 경로 만들어주기-save_root 밑으로 들어감
            gen_img.save(os.path.join(save_root, f'{step}.jpg'))

        ## loss 출력하기
        print(loss_c)
        print(loss_s)
        print(loss_total)
    ##post processing
    ## 이미지 생성하고 저장해주기
    pass

#이 파일이 실행될 때 train_main이 바로 실행될 수 있게끔
if __name__=="__main__":
    train_main()