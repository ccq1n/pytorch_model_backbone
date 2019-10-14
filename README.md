# pytorch_model_integration
The project is based on pytorch and integrates the current mainstream 
network architecture, including VGGnet, ResNet, DenseNet, 
MobileNet and DarkNet (YOLOv2 and YOLOv3).

## Networks Result
| Model       | Params/Million | FLOPs/G  | Time_cast/ms | Top-1 | Top-5 |
| ----------- | -------------- | -------- | ------------ | ----- | ----- |
| --- 2015 --- |
| Vgg11       | 9.738984       | 15.02879  | 205.59       | 70.4  | 89.6  |
| Vgg13       | 9.92388        | 22.45644 | 324.13       | 71.3  | 90.1  |
| Vgg16       | 15.236136      | 30.78787 | 397.33       | 74.4  | 91.9  |
| Vgg19       | 20.548392      | 39.11929 | 451.11       | 74.5  | 92.0  |
| --- 2016 --- |
| ResNet18    | 11.693736      | 3.65921  | 86.56       |       |       |
| ResNet34    | 21.801896      | 7.36109  | 123.07       | 75.81 | 92.6  |
| ResNet50    | 25.557032      | 8.27887  | 293.62       | 77.15 | 93.29 |
| ResNet101   | 44.54916       | 15.71355 | 413.51       | 78.25 | 93.95 |
| ResNet152   | 60.192808      | 23.15064 | 573.09       | 78.57 | 94.29 |
| PreActResNet18    | 11.690792      | 3.65840  | 86.12 |       |       |
| PreActResNet34    | 21.798952      | 7.36029  | 142.51 |       |       |
| PreActResNet50    | 25.545256      | 8.27566  | 296.39 |       |       |
| PreActResNet101   | 44.537384      | 15.71034  | 418.37 |       |       |
| PreActResNet152   | 60.181032      | 23.14743 | 578.81 | 78.90 | 94.50 |
| DarkNet19(YOLOv2)| 8.01556   | 10.90831  | 139.21       |       |       |
| --- 2017 --- |
| DenseNet121 | 7.978734       | 5.69836  | 286.45       |       |       |
| DenseNet169 | 14.149358      | 6.75643  | 375.47       |       |       |
| DenseNet201 | 20.013806      | 8.63084  | 486.14       |       |       |
| DenseNet264 | 33.337582      | 11.57003 | 689.63       |       |       |
| MobileNet   | 4.231976       | 1.14757  | 100.45        | 70.60 |       |
| SqueezeNet  | 1.2524         | 1.69362  | 90.97       | 57.5  | 80.3  |
| SqueezeNet + Simple Bypass   | 1.2524   | 1.69550  | 96.82|60.4| 82.5  |
| SqueezeNet + Complex Bypass  | 1.594928 | 2.40896  |130.98 |58.8| 82.0 |
| --- 2018 --- |
| PeleeNet    | 4.51988        | 4.96656  | 237.18       | 72.6  | 90.6  |
| 1.0-SqNxt-23  |0.690824      | 0.48130  | 69.93        | 59.05 | 82.60 |
| 1.0-SqNxt-23v5|0.909704      | 0.47743  | 58.40        | 59.24 | 82.41 |
| 2.0-SqNxt-23  |2.2474        | 1.12928  | 111.89       | 67.18 | 88.17 |
| 2.0-SqNxt-23v5|3.11524       | 1.12155  | 93.54        | 67.44 | 88.20 |
| MobileNetV2 | 3.56468        | 0.66214  | 138.15       | 74.07 |       |
| DarkNet53(YOLOv3)| 41.609928 | 14.25625 | 275.50       |       |       |
| DLA-34      | 15.784869      | 2.27950 | 70.17        |       |       |
| DLA-46-C    | 1.310885       | 0.40895  | 40.29        | 64.9  | 86.7  |
| DLA-60      | 22.335141      | 2.93399  | 110.80       |       |       |
| DLA-102     | 33.732773      | 4.42848  | 154.27       |       |       |
| DLA-169     | 53.990053      | 6.65083  | 230.39       |       |       |
| DLA-X-46-C  | 1.077925       | 0.37765  | 44.74        | 66.0  | 87.0  |
| DLA-X-60-C  | 1.337765       | 0.40313  | 50.84        | 68.0  | 88.4  |
| DLA-X-60    | 17.650853      | 2.39033  | 131.93       |       |       |
| DLA-X-102   | 26.773157      | 3.58778  | 164.93       |       |       |
| IGCV3-D (0.7) |2.490294 |0.31910|165.14 |68.45| |
| IGCV3-D (1.0) |3.491688 |0.60653|263.80 |72.20| |
| IGCV3-D (1.4) |6.015164 |1.11491|318.40 |74.70| |

input size: (1,3,224,224)

## ImageNet数据准备
### Download
http://www.image-net.org/challenges/LSVRC/2012/downloads

我们需要的是训练集与验证集（等同测试集），一般论文当中只展示验证集上的结果（Top-1 & Top-5）。

 Development kit (Task 1 & 2). 2.5MB. (这个并没有用到)

 Training images (Task 1 & 2). 138GB. MD5: 1d675b47d978889d74fa0da5fadfb00e

 Validation images (all tasks). 6.3GB. MD5: 29b22e2961454d5413ddabcf34fc5622

### 安装
解压下载的数据文件，这可能需要一段时间

    tar xvf ILSVRC2012_img_train.tar -C ./train

    tar xvf ILSVRC2012_img_val.tar -C ./val

对于train数据，解压后是1000个tar文件，需要再次解压，解压脚本dataset/unzip.sh如下

    dir=/data/srd/data/Image/ImageNet/train
    
    for x in `ls $dir/*tar`
    do
        filename=`basename $x .tar`
        mkdir $dir/$filename
        tar -xvf $x -C $dir/$filename
    done
    
    rm *.tar

注：将其中的'dir'修改为自己的文件目录

然后运行

    sh unzip.sh

对于val数据，解压之后是50000张图片，我们需要将每一个类的图片整理到一起，与train一致。将项目dataset/valprep.sh脚本放到val文件夹下运行

    sh valprep.sh

下载好的训练集下的每个文件夹是一类图片，文件夹名对应的标签在下载好的Development kit的标签文件meta.mat中，这是一个matlab文件，scipy.io.loadmat可以读取文件内容，验证集下是50000张图片，每张图片对应的标签在ILSVRC2012_validation_ground_truth.txt中。

数据增强：取图片时随机取，然后将图片放缩为短边为256，然后再随机裁剪224x224的图片，
再把每个通道减去相应通道的平均值，随机左右翻转。