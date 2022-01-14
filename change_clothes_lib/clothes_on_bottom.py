import os
from re import search
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from PIL import Image
import cv2

import json
from numpy import linalg as LA

CLOTHES_DIR = "./clothes_datas/clothes_on_bottom/"

# 四捨五入する関数
def rounding(num):
    integer_num = int(num)
    comparison_value = num+0.5
    if num < comparison_value:
        return math.floor(num)
    else:
        return math.ceil(num)

# 二つのベクトルからなす角を求めて返す。(例えば、人物の骨格情報から得た肩ベクトルと服の骨格情報から得た肩ベクトルとのなす角を求める)
def get_angleFrom2Vec(v,u):
    inner_product = np.dot(v, u)
    n = LA.norm(v) * LA.norm(u)#ベクトルの長さ同士の積
    coscos = inner_product / n
    formed_angle = np.rad2deg(np.arccos(np.clip(coscos, -1.0, 1.0)))
    return(formed_angle)

# 一方のベクトル(rotate_vec)を、1.0度だけ正転させてみて、二つのベクトルとのなす角を得る。
# 1.0度正転させたときのなす角(forward_angle)と1.0度正転させる前でのなす角(angle)を比較して、
# 1.0度正転させたときのなす角の方が大きかったら一方のベクトル(rotate_vec)をなす角だけ回転させたとき、
# なす角が0(平行)になる回転方向は後転なので、signを-1にして返す。
def findDirectionOfRotation(rotate_vec,fixed_vec):
    vec_angle = get_angleFrom2Vec(rotate_vec,fixed_vec)#二つのベクトルのなす角を求める

    c = 1.0 #角度
    if vec_angle+c > 180: #もしなす角+cが180を超えるようなら
        c = (180 - vec_angle)/2 #cをもっと小さい値にする
    
    # このcの角度だけrotate_vecを正転させたforward_line_vecとfixed_vecとのなす角度を求める
    sinsin = math.sin(math.radians(c))
    coscos = math.cos(math.radians(c))
    forward_rot_x = (rotate_vec[0] * sinsin) - (rotate_vec[1] * sinsin)
    forward_rot_y = (rotate_vec[0] * sinsin) + (rotate_vec[1] * coscos)
    forward_line_vec = np.array([forward_rot_x,forward_rot_y])
    forward_angle = get_angleFrom2Vec(forward_line_vec,fixed_vec)
    print("forward_angle="+str(forward_angle))

    # もし正転させたときベクトルとfixed_vecとのなす角がvec_angleより大きいとき、回転方向は後転向きである
    # つまり正転するとなす角は大きくなった場合、なす角でベクトル同士を平行にするには後転する必要がある
    sign = 1 #回転方向を正転にするための角度の符号はプラス
    if forward_angle > vec_angle:
        sign = -1 #回転方向を後転にするための角度の符号はマイナス
    
    return(sign,vec_angle)

# 服(股)画像の横縦幅をh+wの正方形にした画像を作成する※回転した際に服(体)自体が画像からはみ出ないようにするため
# その服(体)画像を肩なす角だけ回転させる
# 回転させた服(体)画像をバウンディングボックスサイズで切り取る
# 人物部位セグム画像を胴体部分のバウンディングボックスサイズを取得する
# 服の体画像を(人物部位セグム画像の胴体の高さ, 服の襟座標を人物部位セグム画像の襟座標に合わさるような横幅)にリサイズする
def adjust_thign_rotate(brank_img,human_segm,thign_img,waist_formed_angle,sign,json_data):
    #１ 服(体)画像の横縦幅をh+wの正方形にした画像を作成する※回転した際に服(体)自体が画像からはみ出ないようにするため
    h,w = thign_img.shape[:2]
    h_l = h+w
    w_l = h+w
    h_l_sa = h_l - h
    w_l_sa = w_l - w
    if h_l_sa%2 == 1:
        h_l += 1
        h_l_sa = h_l - h
    if w_l_sa%2 == 1:
        w_l += 1
        w_l_sa = w_l - w
    sy = int(h_l_sa/2)
    sx = int(w_l_sa/2)

    square_thign = np.zeros((h_l,w_l,4),np.uint8)
    for y in range(h_l):
        for x in range(w_l):
            if y>=sy and y<h+sy and x>=sx and x<w+sx:
                square_thign[y][x] = thign_img[y - sy][x - sx]
    

    #２ その服(体)画像を肩なす角だけ回転させる
    #回転の中心を指定  
    center = (int(w_l/2), int(h_l/2))
    #スケールを指定
    scale = 1.0
    #getRotationMatrix2D関数を使用
    trans = cv2.getRotationMatrix2D(center, sign*waist_formed_angle , scale)
    #アフィン変換
    square_thign = cv2.warpAffine(square_thign, trans, (w_l,h_l))

    '''plt.title("afin")
    plt.imshow(square_torso)
    plt.show()
    plt.title("afin_neck")
    plt.imshow(square_neck_point_mask)
    plt.show()'''


    #３ 回転させた服(体)画像をバウンディングボックスサイズで切り取る
    thign_mask = np.zeros((h_l,w_l,1),np.uint8)
    max_y = 0#服画像の服領域の最大y
    min_y = h_l
    max_x = 0
    min_x = w_l
    for y in range(h_l):
        for x in range(w_l):
            if square_thign[y][x][3] != 0:
                thign_mask[y][x] = 255
                if max_y < y:
                    max_y = y
                if min_y > y:
                    min_y = y
                if max_x < x:
                    max_x = x
                if min_x > x:
                    min_x = x
    print("min_y="+str(min_y))
    print("max_y="+str(max_y))
    print("min_x="+str(min_x))
    print("max_x="+str(max_x))
    print("y="+str(max_y-min_y))
    print("x="+str(max_x-min_x))
    '''plt.title("torso_mask")
    plt.imshow(torso_mask[min_y:max_y,min_x:max_x])
    plt.show()'''


    #４ 人物部位セグム画像を胴体部分のバウンディングボックスサイズを取得する
    human_segm_h,human_segm_w = human_segm.shape[:2]
    human_segm_max_y = 0#服画像の服領域の最大y
    human_segm_min_y = human_segm_h
    human_segm_max_x = 0
    human_segm_min_x = human_segm_w
    for y in range(human_segm_h):
        for x in range(human_segm_w):
            if human_segm[y][x] == 8:
                if human_segm_max_y < y:
                    human_segm_max_y = y+1
                if human_segm_min_y > y:
                    human_segm_min_y = y
                if human_segm_max_x < x:
                    human_segm_max_x = x+1
                if human_segm_min_x > x:
                    human_segm_min_x = x
    print("human_segm_min_y="+str(human_segm_min_y))
    print("human_segm_max_y="+str(human_segm_max_y))
    print("human_segm_min_x="+str(human_segm_min_x))
    print("human_segm_max_x="+str(human_segm_max_x))
    print("y="+str(human_segm_max_y-human_segm_min_y))
    print("x="+str(human_segm_max_x-human_segm_min_x))
    '''plt.title("human_segm")
    plt.imshow(human_segm[human_segm_min_y:human_segm_max_y,human_segm_min_x:human_segm_max_x])
    plt.show()'''
    human_torso = human_segm[human_segm_min_y:human_segm_max_y,human_segm_min_x:human_segm_max_x]#人体部位セグメンテーションの胴体部分
    #result_img = actual_img.copy()#actual_imgをコピーする。これに服を着せる
    #print("result_img.shape="+str(result_img.shape))
    #actual_torso = result_img[human_segm_min_y:human_segm_max_y,human_segm_min_x:human_segm_max_x]#実際の人物画像の胴体部分
    brank_torso = brank_img[human_segm_min_y:human_segm_max_y,human_segm_min_x:human_segm_max_x]#実際の人物画像の胴体範囲のbrank_img
    human_torso_y,human_torso_x = human_torso.shape[:2]


    #５ 服の体画像を(人物部位セグム画像の胴体の高さ, 服の襟座標を人物部位セグム画像の襟座標に合わさるような横幅)にリサイズする
    resize_y = human_segm_max_y - human_segm_min_y
    resize_x = int((neck_max_width_point[0]-neck_min_width_point[0])/(json_data["neck"]["left_point"][0]-json_data["neck"]["right_point"][0])*(max_x-min_x))
    torso_img = square_torso[min_y:max_y,min_x:max_x].copy()
    torso_img = cv2.resize(torso_img, dsize=(resize_x, resize_y))#resizeした服の画像でこれを着せる
    neck_point_mask = square_neck_point_mask[min_y:max_y,min_x:max_x].copy()
    neck_point_mask = cv2.resize(neck_point_mask, dsize=(resize_x, resize_y))
    '''plt.title("resize_torso_img")
    plt.imshow(torso_img)
    plt.show()
    plt.title("resize_neck_point_mask")
    plt.imshow(neck_point_mask)
    plt.show()'''
    
    clothes_min_x = resize_x
    clothes_max_x = 0
    human_min_x = human_segm_max_x
    human_max_x = 0
    for y in range(resize_y):
        clothes_min_x = resize_x#リセットする
        clothes_max_x = 0
        human_min_x = human_segm_max_x
        human_max_x = 0
        print(str(y)+"回")
        for x in range(resize_x):
            if torso_img[y][x][3] != 0:
                if clothes_max_x < x:
                    clothes_max_x = x+1
                if clothes_min_x > x:
                    clothes_min_x = x
        print("clothes_max_x="+str(clothes_max_x))
        print("clothes_min_x="+str(clothes_min_x))
        for foolx in range(human_torso_x):
            if human_segm[human_segm_min_y:human_segm_max_y,human_segm_min_x:human_segm_max_x][y][foolx] == 8:
                if human_max_x < foolx:
                    human_max_x = foolx+1
                if human_min_x > foolx:
                    human_min_x = foolx
        print("human_max_x="+str(human_max_x))
        print("human_min_x="+str(human_min_x))
        brank = torso_img[y:y+1,clothes_min_x:clothes_max_x]
        print(brank)

        brank = cv2.resize(brank,dsize=((human_max_x-human_min_x), 1))#ここで人物部位セグム画像の胴体xサイズにリサイズ
        print(brank)

        #print(actual_torso.shape)
        print(brank_torso.shape)
        for i in range(human_max_x-human_min_x):#ここで人物部位セグム画像の胴体xサイズにリサイズした画像をbrank_imgに貼り付け
            if brank[0][i][3] != 0:
                #actual_torso[y][human_min_x+i] = brank[0][i]
                brank_torso[y][human_min_x+i] = brank[0][i]
    '''plt.title("result")
    plt.imshow(cv2.cvtColor(result_img,cv2.COLOR_BGRA2RGBA))
    plt.show()'''

    return(brank_img)

# 人物の股の部分のmaskを作る※この関数は、人物画像の人物が反転していないときしかうまく動作しない。
def create_thigh_mask(human_segm,waist_vec,waist_point,chest_point):
    h,w = human_segm.shape[:2]
    thigh_mask = np.zeros((h,w,1),np.uint8)
    segm_kernel = [8,9,10]
    # waist_vecからy = a * x を実現する
    ratio = waist_vec[1] / waist_vec[0]#a = ratio
    x = 0
    # thigh_maskの上部線を引く
    # xを正方向にもってく
    while True:
        y = rounding(ratio*x)
        p_x = waist_point[0]+x
        p_y = waist_point[1]+y
        if human_segm[p_y][p_x] in segm_kernel:
            thigh_mask[p_y][p_x] = 255
        else:
            break
        x+=1
    # xを負方向にもってく
    while True:
        y = rounding(ratio*x)
        p_x = waist_point[0]+x
        p_y = waist_point[1]+y
        if human_segm[p_y][p_x] in segm_kernel:
            thigh_mask[p_y][p_x] = 255
        else:
            break
        x-=1
    waist_vec[0]*1+waist_vec[1]*vertical[1] = 0
    sinsin = math.sin(math.radians(90.0))
    coscos = math.cos(math.radians(90.0))
    vertical_rot_x = (waist_vec[0] * sinsin) - (waist_vec[1] * sinsin)
    vertical_rot_y = (waist_vec[0] * sinsin) + (waist_vec[1] * coscos)
    vertical_vec = np.array([vertical_rot_x,vertical_rot_y])
    # thigh_maskの下部線を引く
    for y in range(h):
        for x in range(w):
            if human_segm[y][x] == 8:
                ratio_x = x - waist_point[0]
                ratio_y = rounding(ratio*ratio_x)
                p_x = x
                p_y = waist_point[1] + ratio_y
                if p_y < y:
                    waist_vec == vertical_vec

    

def thigh_change(brank_img,human_segm,candidate,model_name,json_data):
    thigh_img = cv2.imread(CLOTHES_DIR+model_name+"/" + "thigh_"+model_name+".png",-1)

    right_waist_point = np.array([int(candidate[8][0]),int(candidate[8][1])])
    left_waist_point = np.array([int(candidate[11][0]),int(candidate[11][1])])
    waist_vec = left_waist_point - right_waist_point

    clothes_right_waist_point = np.array(json_data["right_waist"])
    clothes_left_waist_point = np.array(json_data["left_waist"])
    clothes_waist_vec = clothes_left_waist_point - clothes_right_waist_point

    sign,waist_formed_angle = findDirectionOfRotation(clothes_waist_vec,waist_vec)

    chest_point = np.array([int(candidate[1][0]),int(candidate[1][1])])

    brank_img = adjust_thign_rotate(brank_img,human_segm,thigh_img,waist_formed_angle,sign,json_data)

    right_sholder_point = np.array([int(candidate[2][0]),int(candidate[2][1])])
    left_sholder_point = np.array([int(candidate[5][0]),int(candidate[5][1])])
    shoulder_vec = left_sholder_point - right_sholder_point
    clothes_right_sholder_point = np.array(json_data["right_shoulder"])
    clothes_left_sholder_point = np.array(json_data["left_shoulder"])
    clothes_shoulder_vec = clothes_left_sholder_point - clothes_right_sholder_point
    shoulder_formed_angle = get_angleFrom2Vec(shoulder_vec,clothes_shoulder_vec)
    print(shoulder_formed_angle)

    neck_max_width_point,neck_min_width_point = search_neck(human_segm)#人物の襟の座標を取得

    neck_point_mask = create_neckPointMask(torso_img,json_data)#服の襟の範囲をプロットした二値画像を作成

    brank_img = adjust_torso_rotate(brank_img,human_segm,torso_img,neck_point_mask,shoulder_formed_angle,json_data,neck_max_width_point,neck_min_width_point)
    return(brank_img)

def foot_change(brank_img,human_segm,candidate,model_name,json_data):
    pass

def bondingCorrection(brank_img,human_segm,model_name):
    pass

def mounting(result_img,brank_img):
    pass

def change(actual_img,human_segm,candidate,model_name,clothes_dir="./clothes_datas/clothes_on_bottom/"):
    global CLOTHES_DIR
    CLOTHES_DIR = clothes_dir
    # 服の骨格情報などを含むjson_dataをロードする
    with open(CLOTHES_DIR+model_name+"/" + model_name+".json") as f:
        json_data = json.load(f)
    # 着せ替えスタート
    result_img = actual_img.copy()

    brank_img = np.zeros(result_img.shape,np.uint8)

    brank_img = thigh_change(brank_img,human_segm,candidate,model_name,json_data)

    brank_img = foot_change(brank_img,human_segm,candidate,model_name,json_data)

    brank_img,boding_img = bondingCorrection(brank_img,human_segm,model_name)
    #cv2.imwrite("./DressApp/dress_lib/images/temporary_imgs/brank_"+model_name+".png",brank_img)
    #cv2.imwrite("./DressApp/dress_lib/images/temporary_imgs/boding_"+model_name+".png",boding_img)

    result_img = mounting(result_img,brank_img)
    plt.title("result")
    plt.imshow(cv2.cvtColor(result_img,cv2.COLOR_BGRA2RGBA))
    plt.show()
    
    return(result_img,brank_img)