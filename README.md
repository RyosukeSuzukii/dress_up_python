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

## Requirements
| 言語/ライブラリ | Version|
| :------------| ---------: |
| python | 3.8.3　|
| django | 3.2.7　|
| docker | 20.10.10 |
| docker-compose |  1.29.2　|
