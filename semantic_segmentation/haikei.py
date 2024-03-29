# 1.各種パッケージをimportします。今回はtorchvisionを使ってsegmentationを行います。
from matplotlib import image
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms


def inference(h,w,img):#推論を行う関数
    # 3.モデルをデバイスに渡し、推論モードに切り替えます。
    print(torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#gpu or cpuへの切り替え

    model_path = './semantic_segmentation/models'
    torch.hub.set_dir(model_path)
    model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
    model = model.to(device)
    model.eval()
    # 4.画像のnumpy配列をtensor型にし、正規化します。また、バッチの次元を追加します。
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0).to(device)
    # 5.推論すると下の右のような画像が得られます。
    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output = output.argmax(0)
    mask = output.byte().cpu().numpy()
    mask = cv2.resize(mask,(w,h))
    img = cv2.resize(img,(w,h))
    return img,mask

def bokasi_filter(img):
    blur = cv2.blur(img,(15,15))
    return blur

def bokasi_compound(img,trimap):
    blur_com = bokasi_filter(img)
    h,w,_ = img.shape
    for y in range(h):
        for x in range(w):
            if trimap[y][x]==255:
                blur_com[y][x] = img[y][x]
    return blur_com


#ヒストグラムを作成する。そして0以外の一番多い画素値のものだけ残して消す
def hist_cut(img,trimap_img):
    trimap = trimap_img.copy()
    hist = np.histogram(img,bins=np.arange(257))
    #print(hist)
    #print(len(hist[1]))
    max = 0
    max_num = 0
    for i in range(len(hist[0])):
        if i != 0 and max_num < hist[0][i]:
            max = i
            max_num = hist[0][i]
    #print(max)
    #print(max_num)
    trimap[img != max] = 0 #これで画素値を消す
    return trimap

#人以外に検出してしまった領域を消す(一番大きい領域だけ残し、飛び地の場所を消す)
def not_human_cut(size_tuple,trimap):
    #print("not_human_cut is trimap = "+str(trimap.dtype))
    zero_img = np.zeros(size_tuple,np.uint8)
    _,cont_img = cv2.threshold(zero_img,10,255,cv2.THRESH_BINARY)#２値化する
    cont_img[trimap>0] = 255
    contours, hierarchy = cv2.findContours(cont_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
    maxs = [0,0]
    for i in range(len(contours)):
        #print(contours[i].shape)
        if maxs[0] < contours[i].shape[0]:
            maxs[0] = contours[i].shape[0]
            maxs[1] = i
    for i in range(len(contours)):
        if i != maxs[1]:
            cv2.drawContours(cont_img, contours, i, 0, -1)

    trimap[cont_img==0] = 0
    return trimap,cont_img

#元画像の人物が画像にぴったり収まるようにtrimapのbouding_boxから切り取る
def bounding_cut(img,trimap):
    #print("trimap of bouding_cut = "+str(trimap.dtype))
    cont_img = np.zeros(trimap.shape,np.uint8)#trimapはfloat64のため変換する必要がある
    cont_img[trimap >= 1] = 255
    contours, hierarchy = cv2.findContours(cont_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    x,y,w,h = cv2.boundingRect(contours[0])
    img = img[y:y+h,x:x+w]
    trimap = trimap[y:y+h,x:x+w]

    #確認用
    '''fig = plt.figure(figsize=(20,9))
    fig.suptitle("plot")
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.title("part_img")
    plt.subplot(1,2,2)
    plt.imshow(trimap,cmap='gray', vmin=0, vmax=255)
    plt.title("part_trimap")
    plt.show()'''

    del cont_img
    del hierarchy
    return img,trimap

# OpenCVで膨張収縮などを行い、trimapを生成する
def gen_trimap(img,mask,k_size=(5,5),ite=1):
    h,w,_ = img.shape
    
    #確認用
    '''plt.title("semantic segmentation")
    plt.imshow(mask)
    plt.show()'''

    #ヒストグラムを作成する。そして0以外の一番多い画素値のものだけ残して消す
    mask = hist_cut(mask,mask)

    #残ってしまったメイン部分以外を消す。つまり一番大きい領域だけ残し、飛び地の領域を消す
    trimap,niti = not_human_cut((h,w),mask)

    #ruslt用
    '''img = cv2.cvtColor(img,cv2.COLOR_BGR2RGBA)
    for y in range(h):
        for x in range(w):
            if trimap[y][x] == 0:
                img[y][x][3] = 0'''

    #膨張収縮処理を行い「前景」「背景」「そのどちらか」に粗く分解したtrimapを生成する
    kernel = np.ones(k_size,np.uint8) #要素が全て1の配列を生成
    eroded = cv2.erode(trimap,kernel,iterations = ite)
    dilated = cv2.dilate(trimap,kernel,iterations = ite)
    trimap = np.full(mask.shape,128)#dtype=float64
    trimap[eroded >= 1] = 255
    trimap[dilated == 0] = 0

    #元画像の人物が画像にぴったり収まるようにtrimapのbouding_boxから切り取る
    img,trimap = bounding_cut(img,trimap)

    return img,trimap,niti

# セマンティックセグメンテーション等の処理を行い、trimapとblur_imgを返す
def cutting_out(dir_path,filename):
    input_dir_name = dir_path
    #下準備
    img = cv2.imread(input_dir_name+filename)
    img = img[...,::-1] #BGR->RGB
    img_h,img_w,_ = img.shape #高さ 幅 色を代入
    #img = cv2.resize(img,(320,320))
    img = cv2.resize(img,(img_w,img_h))#そのままのサイズでリサイズ

    # 推論でセマンティックセグメンテーションmaskを生成する
    img,mask = inference(img_h,img_w,img)

    # maskから人物部分以外を消し、「前景」「背景」「そのどちらか」に分解されているtrimapを生成する
    # また元画像の人物が画像にぴったり収まるようにtrimapのbouding_boxから切り取る
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    img,trimap,niti = gen_trimap(img,mask,k_size=(3,3),ite=2)

    # indexnet_mattingで、より正確に背景を認識してもらうため、
    # trimapの「背景」と「そのどちらか」の領域に該当する人物画像の領域に少しぼかしをかける
    blur_img = bokasi_compound(img,trimap)

    print(blur_img.shape)
    print(trimap.shape)
    #確認用
    '''plt.title("trimap")
    plt.imshow(trimap)
    plt.show()'''
    return(blur_img,trimap)
