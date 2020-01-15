# pytorch_model_integration
The project is based on pytorch and integrates the current mainstream 
network architecture, including VGGnet, ResNet, DenseNet, 
MobileNet and DarkNet (YOLOv2 and YOLOv3).

## Networks Result
| Model       | Params/Million | FLOPs/G  | Time_cast/ms | Top-1 | Top-5 |
| ----------- | -------------- | -------- | ------------ | ----- | ----- |
| --- **2015** --- |
| Vgg11       | 9.738984       | 15.02879  | 205.59       | 70.4  | 89.6  |
| Vgg13       | 9.92388        | 22.45644 | 324.13       | 71.3  | 90.1  |
| Vgg16       | 15.236136      | 30.78787 | 397.33       | 74.4  | 91.9  |
| Vgg19       | 20.548392      | 39.11929 | 451.11       | 74.5  | 92.0  |
| --- **2016** --- |
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
| --- **2017** --- |
| DenseNet121 | 7.978734       | 5.69836  | 286.45       |       |       |
| DenseNet169 | 14.149358      | 6.75643  | 375.47       |       |       |
| DenseNet201 | 20.013806      | 8.63084  | 486.14       |       |       |
| DenseNet264 | 33.337582      | 11.57003 | 689.63       |       |       |
|ResNeXt50_2x40d|25.425|8.29756|364.24|77.00||
|ResNeXt50_4x24d|25.292968|8.37150|416.01|77.40||
|ResNeXt50_8x14d|25.603016|8.58994|444.33|77.70||
|ResNeXt50_32x4d|25.028904 | 8.51937|460.20|77.80||
|ResNeXt101_2x40d|44.456296|15.75783|640.83|78.3||
|ResNeXt101_4x24d| 44.363432| 15.84712|627.48|78.6||
|ResNeXt101_8x14d|45.104328|16.23445|870.31|78.7||
|ResNeXt101_32x4d|44.177704|16.02570|952.88|78.8||
| MobileNet   | 4.231976       | 1.14757  | 100.45        | 70.60 |       |
| SqueezeNet  | 1.2524         | 1.69362  | 90.97       | 57.50  | 80.30  |
| SqueezeNet + Simple Bypass   | 1.2524   | 1.69550  | 96.82|60.40| 82.50  |
| SqueezeNet + Complex Bypass  | 1.594928 | 2.40896  |130.98 |58.80| 82.00 |
| --- **2018** --- |
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
| --- **2019** --- |
| EfficientNet-B0 |5.288548 |0.01604 |186.61 |76.30|93.20|
| EfficientNet-B1 |7.794184 |0.02124 |266.05 |78.80|94.40|
| EfficientNet-B2 |9.109994 |0.02240 |277.94 |79.80|94.90|
| EfficientNet-B3 |12.233232 |0.02905 | 376.24 |81.10|95.50|
| EfficientNet-B4 |19.341616 |0.03762 |513.91 |82.60|96.30|
| EfficientNet-B5 |30.389784 |0.05086 |721.95 |83.30|96.70|
| EfficientNet-B6 |43.040704 |0.06443 |1062.64 |84.00|96.90|
| EfficientNet-B7 |66.34796 |0.08516 |1520.88 |84.40|97.10|

### GoogleNet Inception V1-V4

| Model       | Params/Million | FLOPs/G  | Time_cast/ms | Top-1 | Top-5 |
| ----------- | -------------- | -------- | ------------ | ----- | ----- |
| --- **2014** --- |
| GoogleNet V1 |6.998552 |3.20387   |85.95|||
| GoogleNet V1 (LRN)|6.998552 |3.20387   |192.64|71.00|90.80|
| GoogleNet V1 (Bn)|7.013112|3.21032 |139.42|73.20||
| --- **2015** --- |
| GoogleNet V2|11.204936|4.08437   |127.71|76.60||
| GoogleNet V3|23.834568|7.60887   |208.01|78.80|94.40|
| --- **2016** --- |
| GoogleNet V4|42.679816|12.31977  |324.36|80.00|95.10|

Note: GoogleNet V1 does not include the Bn layer, but after the first two layers of convolution, LocalResponseNorm is added, 
this operation will increase the calculation time of the model. So we found that GoogleNet V1 is slower than GoogleNet V1_Bn.

For Time_cast, we set the input size: (1, 3, 224, 224), and then test multiple rounds of averaging 
(time is susceptible to interference from CPU operating state).

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

## CROP-1, CROP-5 and CROP-10 test methods for ImageNet validation set.

Due to the existence of a fully connected layers, we neeed to limit the size of the images in the input network. Set the size of the image input neural network to 224x224, but the size of the image in the test set is not fixed. It is difficult to completely cover the information of the target object in the image by only the center clipping method, so we crop the image at multiple locations.

One-crop of an image is created by cropping one 224 × 224 regions from the center of a 256 × 256 image; Five-crop is five 224 × 224 sized image regions cropped from top left, top right, bottom left, bottom right and center of original image; Ten-crop is horizontally flipping each cropped region base on the results of five-crop.

Use Pytorch,
