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

## 条件

## Licence   
### indexnet_matting licence
> `IndexNet Matting for non-commercial purposes

Copyright (c) 2019 Hao Lu All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.`
