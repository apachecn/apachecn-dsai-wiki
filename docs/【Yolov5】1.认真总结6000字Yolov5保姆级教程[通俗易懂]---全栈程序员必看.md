<!--yml
category: 深度学习
date: 2023-01-30 22:34:43
-->

# 【Yolov5】1.认真总结6000字Yolov5保姆级教程[通俗易懂] - 全栈程序员必看

> 来源：[https://javaforall.cn/126760.html](https://javaforall.cn/126760.html)

大家好，又见面了，我是你们的朋友全栈君。

**目录**

[一、前言](#main-toc)

[二、学习内容](#%E4%B8%80%E3%80%81%E5%AD%A6%E4%B9%A0%E5%86%85%E5%AE%B9)

[二、版本与配置声明](#%E4%BA%8C%E3%80%81%E7%89%88%E6%9C%AC%E4%B8%8E%E9%85%8D%E7%BD%AE%E5%A3%B0%E6%98%8E)

[三、Yolov5的准备](#%E4%B8%89%E3%80%81Yolov5%E7%9A%84%E5%87%86%E5%A4%87)

[1.下载Yolov5](#1.%E4%B8%8B%E8%BD%BDYolov5)

[2.安装依赖库](#2.%E5%AE%89%E8%A3%85%E4%BE%9D%E8%B5%96%E5%BA%93)

[3.运行检测](#3.%E8%BF%90%E8%A1%8C%E6%A3%80%E6%B5%8B)

[四、训练集](#%E5%9B%9B%E3%80%81%E8%AE%AD%E7%BB%83%E9%9B%86)

[五、制作标签](#%E4%BA%94%E3%80%81%E5%88%B6%E4%BD%9C%E6%A0%87%E7%AD%BE)

[1.下载labelme](#1.%E4%B8%8B%E8%BD%BDlabelme)

[2.安装依赖库](#2.%E5%AE%89%E8%A3%85%E4%BE%9D%E8%B5%96%E5%BA%93)

[3.labelme操作](#3.labelme%E6%93%8D%E4%BD%9C)

[ 4.json转txt](#%C2%A04.json%E8%BD%ACtxt)

[五、修改配置文件](#%E4%BA%94%E3%80%81%E4%BF%AE%E6%94%B9%E9%85%8D%E7%BD%AE%E6%96%87%E4%BB%B6)

[1.coco128.yaml](#1.coco128.yaml)

[2.yolov5配置](#2.yolov5%E9%85%8D%E7%BD%AE)

[ 六、训练train](#%C2%A0%E5%85%AD%E3%80%81%E8%AE%AD%E7%BB%83train)

[七、识别detect](#%E4%B8%83%E3%80%81%E8%AF%86%E5%88%ABdetect)

[八、debug](#%E5%85%AB%E3%80%81debug)

[九、百度网盘资源](#%E5%85%AB%E3%80%81%E7%99%BE%E5%BA%A6%E7%BD%91%E7%9B%98%E8%B5%84%E6%BA%90)

[十、结语](#%E4%B9%9D%E3%80%81%E7%BB%93%E8%AF%AD)

* * *

# 一、前言

1.集成的资源我放在了文末，包括我自己做成的成品，可以直接train与detect。我发在百度网盘上。

![【Yolov5】1.认真总结6000字Yolov5保姆级教程[通俗易懂]](img/41a2b1236e2dd4673d39b9ebdd89c511.png)

 2.我本人没有学过深度学习，我只是在做视觉项目的时候记录了过程，主要是能够让读者复现，直接使用，而且我不讲原理。如果深入想了解yolov5的原理，可以去看热度比较高的博主做的

3.如果有问题可以在评论区里讨论，或者私信我都行，提问前请先点赞支持一下博主^_^。

# 二、学习内容

2020年6月25日，Ultralytics发布了YOLOV5 的第一个正式版本，其性能与YOLO V4不相伯仲，同样也是现今最先进的对象检测技术，并在推理速度上是目前最强，yolov5按大小分为四个模型yolov5s、yolov5m、yolov5l、yolov5x。

今天我们来学习一下如何简单使用这个算法

文章特点：一个完整的流程，从头教到尾，不讲冗长的理论，实操，看完本篇文章，训练与识别都是没有问题的，我以王者荣耀作为训练集，可以先看看效果

[Yolov5展示视频（b站），可以直接戳这个也可以看下面俩](https://www.bilibili.com/video/BV1Qg41177G4/ "Yolov5展示视频（b站），可以直接戳这个也可以看下面俩")

以下是操作的流程图

# ![【Yolov5】1.认真总结6000字Yolov5保姆级教程[通俗易懂]](img/0498ebaa22413011cc1eb52e9fa5fa21.png)

# 二、版本与配置声明

> 官方要求python>=3.7,Pytorch>=1.5
> 
> 我的：python                            3.7.1
> 
> ———————————————————-Yolov5需要
> 
> # base ————————————–
> matplotlib>=3.2.2
> numpy>=1.18.5
> opencv-python>=4.1.2
> Pillow
> PyYAML>=5.3.1
> scipy>=1.4.1
> torch>=1.7.0
> torchvision>=0.8.1
> tqdm>=4.41.0
> 
> # logging ————————————–
> tensorboard>=2.4.1
> # wandb
> 
> # plotting ————————————–
> seaborn>=0.11.0
> pandas
> 
> # export ————————————–
> # coremltools>=4.1
> # onnx>=1.9.0
> # scikit-learn==0.19.2  # for coreml quantization
> 
> # extras ————————————–
> # Cython  # for pycocotools https://github.com/cocodataset/cocoapi/issues/172
> # pycocotools>=2.0  # COCO mAP
> # albumentations>=1.0.2
> thop  # FLOPs computation
>  
> 
> ———————————————————–labelme需要
> 
> PyQt5                              5.15.4
> 
> labelme                            4.5.9
> 
> ———————————————————-

联想小新Air 15

![【Yolov5】1.认真总结6000字Yolov5保姆级教程[通俗易懂]](img/e66600987eea29c478e9e01dbddbf584.png)

# 三、Yolov5的准备

## 1.下载Yolov5

[https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5 "https://github.com/ultralytics/yolov5")，放在合理的位置，如果这个下的慢的话见文末资源

## 2.安装依赖库

“版本声明”中的库需要安装的，主要的是这几个

requests

pandas

pyyaml

matplotlib

seaborn

cython

numpy

tensorboard

大部分都能pip install 。重点说两个

（1）对于Pytorch，在Anaconda Prompt里输入

pip install torch==1.7.0+cu101 torchvision==0.8.1+cu101 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

（2）对于wandb，[wandb安装方法](https://blog.csdn.net/hhhhhhhhhhwwwwwwwwww/article/details/116124285?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522162791597216780265438950%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=162791597216780265438950&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_v2~rank_v29-8-116124285.first_rank_v2_pc_rank_v29&utm_term=wandb&spm=1018.2226.3001.4187 "wandb安装方法")，这个好像不是必须的，但我还是下了，版本为0.10.11，刚好能兼容，作用就是对训练分析，如图所示

![【Yolov5】1.认真总结6000字Yolov5保姆级教程[通俗易懂]](img/e66600987eea29c478e9e01dbddbf584.png)

## 3.运行检测

下载完yolov5后，点detect，运行

![【Yolov5】1.认真总结6000字Yolov5保姆级教程[通俗易懂]](img/e66600987eea29c478e9e01dbddbf584.png)

这个是帮你检测能不能正常运行的

若正常：

```
D:\Anaconda\python.exe C:/Users/86189/Desktop/yolov5-master/yolov5-master/detect.py
detect: weights=yolov5s.pt, source=data/images, imgsz=640, conf_thres=0.25, iou_thres=0.45, max_det=1000, device=, view_img=False, save_txt=False, save_conf=False, save_crop=False, nosave=False, classes=None, agnostic_nms=False, augment=False, visualize=False, update=False, project=runs/detect, name=exp, exist_ok=False, line_thickness=3, hide_labels=False, hide_conf=False, half=False
YOLOv5  2021-7-17 torch 1.7.0+cu101 CUDA:0 (GeForce MX350, 2048.0MB)

Downloading https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt to yolov5s.pt...
100%|██████████| 14.1M/14.1M [01:32<00:00, 160kB/s]

Fusing layers... 
Model Summary: 224 layers, 7266973 parameters, 0 gradients
image 1/2 C:\Users189\Desktop\yolov5-master\yolov5-master\data\images\bus.jpg: 640x480 4 persons, 1 bus, 1 fire hydrant, Done. (0.055s)
image 2/2 C:\Users189\Desktop\yolov5-master\yolov5-master\data\images\zidane.jpg: 384x640 2 persons, 2 ties, Done. (0.024s)
Results saved to runs\detect\exp
Done. (0.197s)

Process finished with exit code 0 
```

在runs中能发现被处理过的标签，说明成功了！

![【Yolov5】1.认真总结6000字Yolov5保姆级教程[通俗易懂]](img/e66600987eea29c478e9e01dbddbf584.png)

若程序报错，大概率是因为有的库版本不正确或者还未安装，这个自己调试一下即可，应该没有太大难度 

# 四、训练集

训练集至少50张起步才有效果

训练集就是你需要train并用于detect的东西，我以王者荣耀作为例子，你可以跟着我来一遍，资源在文末。要做自己的训练集的话再看第五步。跟着我的话可以不用做标签，因为资源中已经做好了

如下图所示创建文件夹，让操作更清晰方便

images就是训练集的图片，labels就是训练集的标签，train的话是用于训练的，test就是用于测试的

![【Yolov5】1.认真总结6000字Yolov5保姆级教程[通俗易懂]](img/e66600987eea29c478e9e01dbddbf584.png)

# 五、制作标签

## 1.下载labelme

[https://github.com/wkentaro/labelme](https://github.com/wkentaro/labelme "https://github.com/wkentaro/labelme")，如果下载得慢的话见文末资源

![【Yolov5】1.认真总结6000字Yolov5保姆级教程[通俗易懂]](img/e66600987eea29c478e9e01dbddbf584.png)

点Download Zip，下载后找到该文件，解压，无需配置环境变量 

## 2.安装依赖库

在Anaconda Prompt里pip install pyqt5和pip install labelme

## 3.labelme操作

然后在Anaconda Prompt里输入labelme，打开界面如下

![【Yolov5】1.认真总结6000字Yolov5保姆级教程[通俗易懂]](img/e66600987eea29c478e9e01dbddbf584.png)

 可以选择打开一个文件或者文件夹，如果是打开文件夹的话就会是下面那样子

右击，点击rectangle，即画矩形框，框选你要识别训练的东西，举王者荣耀的例子

![【Yolov5】1.认真总结6000字Yolov5保姆级教程[通俗易懂]](img/e66600987eea29c478e9e01dbddbf584.png)

 框选之后输入标签的名字，注意，可以框选多个作为标签。框选完一张图后保存，然后接着下一张图。保存的文件格式是.json

##  4.json转txt

由于yolov5只认txt而不认json，因此还要有一个转换的过程

在yolov5-master中创建一个.py文件，代码如下

```
import json
import os

name2id =  {'hero':0,'sodier':1,'tower':2}#标签名称

def convert(img_size, box):
    dw = 1\. / (img_size[0])
    dh = 1\. / (img_size[1])
    x = (box[0] + box[2]) / 2.0 - 1
    y = (box[1] + box[3]) / 2.0 - 1
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def decode_json(json_floder_path, json_name):
    txt_name = 'C:\\Users\189\\Desktop\\' + json_name[0:-5] + '.txt'
    #存放txt的绝对路径
    txt_file = open(txt_name, 'w')

    json_path = os.path.join(json_floder_path, json_name)
    data = json.load(open(json_path, 'r', encoding='gb2312',errors='ignore'))

    img_w = data['imageWidth']
    img_h = data['imageHeight']

    for i in data['shapes']:

        label_name = i['label']
        if (i['shape_type'] == 'rectangle'):
            x1 = int(i['points'][0][0])
            y1 = int(i['points'][0][1])
            x2 = int(i['points'][1][0])
            y2 = int(i['points'][1][1])

            bb = (x1, y1, x2, y2)
            bbox = convert((img_w, img_h), bb)
            txt_file.write(str(name2id[label_name]) + " " + " ".join([str(a) for a in bbox]) + '\n')

if __name__ == "__main__":

    json_floder_path = 'C:\\Users\189\\Desktop\\哈哈哈\\'
    #存放json的文件夹的绝对路径
    json_names = os.listdir(json_floder_path)
    for json_name in json_names:
        decode_json(json_floder_path, json_name) 
```

标注地方是需要修改的，有几个标签名就写几个标签名，而且这是一个文件夹里所有的json一起转化，存放txt的路径改为labels的train中（还记得下面这张图吗）

![【Yolov5】1.认真总结6000字Yolov5保姆级教程[通俗易懂]](img/e66600987eea29c478e9e01dbddbf584.png)

![【Yolov5】1.认真总结6000字Yolov5保姆级教程[通俗易懂]](img/e66600987eea29c478e9e01dbddbf584.png)

 转化完后大概会是这样子，如果一张图有多个标签的话，这个数据就会变多

# 五、修改配置文件

![【Yolov5】1.认真总结6000字Yolov5保姆级教程[通俗易懂]](img/e66600987eea29c478e9e01dbddbf584.png)

## 1.coco128.yaml

先复制一份，可以粘贴到my_dates中，改名为**mydata（当然你想改啥名字，想放哪里都行，但是要记住路径记住名字，2.也一样）**

![【Yolov5】1.认真总结6000字Yolov5保姆级教程[通俗易懂]](img/e66600987eea29c478e9e01dbddbf584.png)

 注意这个mydata的路径，不要放错了；此外，mydata1_yaml应该是训练后自动生成的，不用管。

mydata.yaml文件需要修改的参数是nc与names。nc是标签名个数，names就是标签的名字，王者荣耀的例子中有10个标签，标签名字都如下。同时要把path注释掉

![【Yolov5】1.认真总结6000字Yolov5保姆级教程[通俗易懂]](img/e66600987eea29c478e9e01dbddbf584.png)

*（注意path那里要注释掉，评论区 Hariod 兄弟说原先是没有被注释的）*

还需要修改一个路径，注意这里是存放训练集图片的相对路径，斜杠别搞反了，相对路径就是对于yolov5-master的路径，我的如下（这就是我为什么建议创建文件夹的原因）

![【Yolov5】1.认真总结6000字Yolov5保姆级教程[通俗易懂]](img/e66600987eea29c478e9e01dbddbf584.png)

## 2.yolov5配置

yolov5有4种配置，不同配置的特性如下，我这里选择yolov5s，速度最快，但是效果最拉胯

![【Yolov5】1.认真总结6000字Yolov5保姆级教程[通俗易懂]](img/e66600987eea29c478e9e01dbddbf584.png)

先复制一份yolov5s

![【Yolov5】1.认真总结6000字Yolov5保姆级教程[通俗易懂]](img/e66600987eea29c478e9e01dbddbf584.png)

可以粘贴到my_dates中，改名为**mydata_1**,需要修改的参数是nc，nc就是标签的数量，王者荣耀的例子是10个，故改成10

![【Yolov5】1.认真总结6000字Yolov5保姆级教程[通俗易懂]](img/e66600987eea29c478e9e01dbddbf584.png)

#  六、训练train

 1.train.py

打开这个文件，需要修改的参数比较多

第一个是with open，参数要加上encoding=’utf-8’，不然的话很可能会出现编码报错UnicodeDecodeError: ‘gbk‘ codec can‘t decode byte 0xad in position 577

![【Yolov5】1.认真总结6000字Yolov5保姆级教程[通俗易懂]](img/e66600987eea29c478e9e01dbddbf584.png)

以下修改的都是default处 

第二个是484行的配置，用的是yolov5s

第三个是486行和488行的路径

第四个是491行，因为我的电脑拉胯，所以不能按照原配置搞（这就是我为什么在一开始就声明配置的原因）

第五个是495行，原来长宽都是640，不行的话减32直至可以run

第六个是513行

这一大串参数都是根据我的低配联想小新来的，我按照这样是能run的，游戏本的话配置还能再调高些

![【Yolov5】1.认真总结6000字Yolov5保姆级教程[通俗易懂]](img/e66600987eea29c478e9e01dbddbf584.png)

![【Yolov5】1.认真总结6000字Yolov5保姆级教程[通俗易懂]](img/e66600987eea29c478e9e01dbddbf584.png)

这个时间很长！我当时试过如果是5张图片，要花5分钟左右，如果是王者荣耀这个例子106张图，大概要1个小时

![【Yolov5】1.认真总结6000字Yolov5保姆级教程[通俗易懂]](img/e66600987eea29c478e9e01dbddbf584.png)

*（train效果如上图所示）*

在runs\train\exp\weights中能看到有一个best和last，意思是最好的和最近的

# 七、识别detect

如下图所示是需要修改的参数，也是best和测试图片的相对路径，不要搞错了。结果在runs\detect\exp中看

![【Yolov5】1.认真总结6000字Yolov5保姆级教程[通俗易懂]](img/e66600987eea29c478e9e01dbddbf584.png)

我的测试是个视频（视频也可图片也可），出来的结果就是文章开头放的视频

效果不是很好，但是还是能分辨个大概，原因有两个

一是因为用了最拉的yolov5s，二是训练的文件还不够多

![【Yolov5】1.认真总结6000字Yolov5保姆级教程[通俗易懂]](img/e66600987eea29c478e9e01dbddbf584.png)

*detect效果如图所示，画红线的地方为识别出来的物体*

# 八、debug

我猜测大多数问题为：

1.wandb报错。安装方法的链接已经在文中了。如果要使用wandb的话需要注册那个网站，然后他会给个码给你，复制后你在控制台里粘贴，然后才能用（大概是这样）。

2.文件路径没写对。有点惭愧，文件路径我搞的比较乱（自我吐槽），大家要注意。正因如此我做了个框架图。

3.显卡爆了，那就调低train中我列出来的那几行default

4.有朋友说他在训练时，box obj cls labels的值为0或nan。正常情况下是正常的数（我发了训练的时候的图片），我猜测可能是训练集标签没做好 或者 路径没写对 或者 default没调好

5.建议路径为全英文，不要带中文，否则可能会出现意料之外的错误

# 九、百度网盘资源

链接：[https://pan.baidu.com/s/1YmZOPzcVaA0TuupMDW93SQ ](https://pan.baidu.com/s/1YmZOPzcVaA0TuupMDW93SQ%C2%A0 "https://pan.baidu.com/s/1YmZOPzcVaA0TuupMDW93SQ ")
提取码：vhw1 

# 十、结语

*   我也是个小白，可能存在很多不足之处，希望有不足之处可以包容，我会改正的(⸝⸝•‧̫•⸝⸝)
*   最后感谢我的hxd，很多都是他教的，我自己尝试过一遍并且成功了，整理排版才的来这篇文章
*   如果遇到出现错误的，自己先多找一找问题，能力在debug中会不断提高的

你们的每个赞都能让我开心好几天✿✿ヽ(°▽°)ノ✿

版权声明：本文内容由互联网用户自发贡献，该文观点仅代表作者本人。本站仅提供信息存储空间服务，不拥有所有权，不承担相关法律责任。如发现本站有涉嫌侵权/违法违规的内容， 请发送邮件至 举报，一经查实，本站将立刻删除。

发布者：全栈程序员栈长，转载请注明出处：https://javaforall.cn/126760.html原文链接：https://javaforall.cn

**【正版授权，激活自己账号】：** [Jetbrains全家桶Ide使用，1年售后保障，每天仅需1毛](https://mh5ittqva6.feishu.cn/docs/doccnA8l3EZCT2ILuQrlNX0XVif "正版授权 一人一码")

**【官方授权 正版激活】：** [官方授权 正版激活 支持Jetbrains家族下所有IDE 使用个人JB账号...](https://mh5ittqva6.feishu.cn/docs/doccnA8l3EZCT2ILuQrlNX0XVif "官方授权 正版激活")