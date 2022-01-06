# dress_up_python
## 正面全身人物画像から試着
正面向きの全身人物画像から背景を切り抜き、人物の部位ごとの分類と骨格情報を取得を行う。  
そして、その背景削除人物画像、人物部位セグメンテーション画像、骨格情報、用意された衣類画像,衣類Jsonから
衣類を背景削除人物画像に重ね、試着を行います。

## 利用したリポジトリ
### A simple and minimal bodypix inference in python
> poppinace / indexnet_matting  
> <https://github.com/poppinace/indexnet_matting>
### indexnet_matting
> ajaichemmanam / simple_bodypix_python  
> <https://github.com/ajaichemmanam/simple_bodypix_python>
### Body and Hand Pose Estimation
> Hzzone / pytorch-openpose  
> <https://github.com/Hzzone/pytorch-openpose>

## Requirements 動作環境
| 言語/ライブラリ | Version|
| :------------| ---------: |
| Python | 3.7.2　|
| torch | 1.10.0　|
| torchvision |  0.11.1　|
| opencv-python | 4.5.3.56 |
| numpy |  1.21.3　|
| Pillow |  8.4.0 |
| matplotlib |  3.4.3　|
| scipy |  1.7.1　|
| scikit-image |  0.18.3　|
| tensorflow |  2.7.0　|
| tensorflowjs |  3.11.0　|
| tqdm |  4.19.9　|
