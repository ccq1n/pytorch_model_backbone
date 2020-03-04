'''
Calculate the number of parameters, time cast, and FLOPs of different networks.
'''
from thop import profile
from models import *
import time
import csv

annotations_new = {
    # 'GoogleNetV1':GoogleNetV1(),
    # 'GooglrNetV1_Bn':GoogleNetV1_Bn(),
    # 'GoogleNetV2':GoogleNetV2(),
    # 'GoogleNetV3':GoogleNetV3(),
    # 'GoogleNetV4':GoogleNetV4(),
    # 'EfficientNetB0':EfficientNet_B0(),
    # 'EfficientNetB1':EfficientNet_B1(),
    # 'EfficientNetB2':EfficientNet_B2(),
    # 'EfficientNetB3':EfficientNet_B3(),
    # 'EfficientNetB4':EfficientNet_B4(),
    # 'EfficientNetB5':EfficientNet_B5(),
    # 'EfficientNetB6':EfficientNet_B6(),
    # 'EfficientNetB7':EfficientNet_B7(),
    # 'ResNeXt50_2x40d': ResNeXt50(2, 40),
    # 'ResNeXt50_4x24d': ResNeXt50(4, 24),
    # 'ResNeXt50_8x14d': ResNeXt50(8, 14),
    # 'ResNeXt50_32x4d': ResNeXt50(32, 4),
    # 'ResNeXt101_2x40d': ResNeXt101(2, 40),
    # 'ResNeXt101_4x24d': ResNeXt101(4, 24),
    # 'ResNeXt101_8x14d': ResNeXt101(8, 14),
    # 'ResNeXt101_32x4d': ResNeXt101(32, 4),
    # 'VGG11': VGG('VGG11'),
    # 'VGG13': VGG('VGG13'),
    # 'VGG16': VGG('VGG16'),
    # 'VGG19': VGG('VGG19'),
    'ResNet18': ResNet18(),
    'ResNet34': ResNet34(),
    # 'ResNet50': ResNet50(),
    # 'ResNet101': ResNet101(),
    # 'ResNet152': ResNet152(),
    # 'MobileNetV2': MobileNetV2(),
    # 'DarkNet53': DarkNet_53(),
    # 'DarkNet19': DarkNet_19(),
    # 'MobileNet': MobileNet(),
    # 'PeleeNet': PeleeNet(),
    'DenseNet121': DenseNet121(),
    'DenseNet169': DenseNet169(),
    'DenseNet201': DenseNet201(),
    'DenseNet264': DenseNet264(),
    'DenseNet161': DenseNet161(),
    # 'DLANet34': DLANet34(),
    # 'DLANet46_C': DLANet46_C(),
    # 'DLANet60': DLANet60(),
    # 'DLANet102': DLANet102(),
    # 'DLANet169': DLANet169(),
    # 'DLANetX46_C': DLANetX46_C(),
    # 'DLANetX60_C': DLANetX60_C(),
    # 'DLANetX60': DLANetX60(),
    # 'DLANetX102': DLANetX102(),
    # 'SqueezeNet': SqueezeNet(),
    # 'SqueezeNet_Simple': SqueezeNet_Simple(),
    # 'SqueezeNet_Complex': SqueezeNet_Complex(),
    # '1.0-SqNxt-23': SqNxt_x1_23(),
    # '1.0-SqNxt-23v5': SqNxt_x1_23v5(),
    # '2.0-SqNxt-23': SqNxt_x2_23(),
    # '2.0-SqNxt-23v5': SqNxt_x2_23v5(),
    # '0.7-igcv3': IGCV3(0.7),
    # '1.0-igcv3': IGCV3(),
    # '1.4-igcv3': IGCV3(1.4),
    # 'PreActResNet18': PreActResNet18(),
    # 'PreActResNet34': PreActResNet34(),
    # 'PreActResNet50': PreActResNet50(),
    # 'PreActResNet101': PreActResNet101(),
    # 'PreActResNet152': PreActResNet152(),
    # 'DPN92': DPN92(),
    # 'DPN98': DPN98()
}

annotations = {
               'SENet18': SENet18(),
               'ShuffleNetG2': ShuffleNetG2(),
               'ShuffleNetG3': ShuffleNetG3(),
               'ShuffleNetV2': ShuffleNetV2(1),
}

def calculate_params_scale(model, name='model', format=''):
    scale = 0
    if isinstance(model, torch.nn.Module):
        # method 1
        scale = sum([param.nelement() for param in model.parameters()])
        # model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        # scale = sum([np.prod(p.size()) for p in model_parameters])
    elif isinstance(model, OrderedDict):
        # method 3
        for key, val in model.items():
            if not isinstance(val, torch.Tensor):
                continue
            scale += val.numel()
    if format == 'million':  # (百万)
        scale /= 1000000
        print("\n*** [%s] Number of params: " % name + str(scale) + '\tmillion...')
        return scale
    else:
        print("\n*** [%s] Number of params: " + str(scale) + '\t...')
        return scale

def calculate_FLOPs_scale(model, inputs, multiply_adds=False, use_gpu=False):
    """
    forked from FishNet @ github
    https://github.com/kevin-ssy/FishNet/blob/master/utils/profile.py
    another: https://github.com/Lyken17/pytorch-OpCounter
    no bias: K^2 * IO * HW
    multiply_adds : False in FishNet Paper, but True in DenseNet paper
    """
    assert isinstance(model, torch.nn.Module)
    USE_GPU = use_gpu and torch.cuda.is_available()
    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups) * (
            2 if multiply_adds else 1)
        bias_ops = 1 if self.bias is not None else 0
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width
        list_conv.append(flops)
    def deconv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups) * (
            2 if multiply_adds else 1)
        bias_ops = 1 if self.bias is not None else 0
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width
        list_deconv.append(flops)
    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1
        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()
        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)
    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement())
    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())
    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width
        list_pooling.append(flops)
    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.ConvTranspose2d):
                net.register_forward_hook(deconv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                net.register_forward_hook(pooling_hook)
            return
        for c in childrens:
            foo(c)
    multiply_adds = multiply_adds
    list_conv, list_deconv, list_bn, list_relu, list_linear, list_pooling = [], [], [], [], [], []
    foo(model)
    if USE_GPU:
        inputs = inputs.cuda()
        model = model.cuda()
    _ = model(inputs)
    total_flops = (sum(list_conv) + sum(list_deconv) + sum(list_linear)
                   + sum(list_bn) + sum(list_relu) + sum(list_pooling))
    print('  + Number of FLOPs: %.5fG' % (total_flops / 1e9 / 2))
    return '%.5f' % (total_flops / 1e9 / 2)

def calculate_layers_num(model, layers=('conv2d', 'classifier')):
    assert isinstance(model, torch.nn.Module)
    type_dict = {'conv2d': torch.nn.Conv2d,
                 'bnorm2d': torch.nn.BatchNorm2d,
                 'relu': torch.nn.ReLU,
                 'fc': torch.nn.Linear,
                 'classifier': torch.nn.Linear,
                 'linear': torch.nn.Linear,
                 'deconv2d': torch.nn.ConvTranspose2d}
    nums_list = []
    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, type_dict[layer]):
                pass
            return 1
        count = 0
        for c in childrens:
            count += foo(c)
        return count
    def foo2(net, layer):
        count = 0
        for n, m in net.named_modules():
            if isinstance(m, type_dict[layer]):
                count += 1
        return count
    for layer in layers:
        # nums_list.append(foo(model))
        nums_list.append(foo2(model, layer))
    total = sum(nums_list)
    strtip = ''
    for layer, nums in zip(list(layers), nums_list):
        strtip += ', %s: %s' % (layer, nums)
    print('  + Number of layers: %s %s ...' % (total, strtip))
    return total

def calculate_time_cost(model, inputs, toc=1, use_gpu=False, pritout=False):
    if not use_gpu:
        tic, toc = time.time(), toc
        y = [model(inputs) for _ in range(toc)][0]
        toc = (time.time() - tic) / toc
        toc = toc * 1000
        # print('  + time cost: %.2f ms\t' % toc)
        # if not isinstance(y, (list, tuple)):
        #     y = [y]
        # if pritout:
        #     print('  + preditions: %s xfc.' % len(y), [yy.max(1) for yy in y])
        return toc
    else:
        assert torch.cuda.is_available()
        model, x = model.cuda(), inputs.cuda()
        tic, toc = time.time(), toc
        y = [model(x) for _ in range(toc)][0]
        toc = (time.time() - tic) / toc
        toc = toc * 1000
        # print('  + time cost: %.2f ms\t' % toc)
        # if not isinstance(y, (list, tuple)):
        #     y = [y]
        # if pritout:
        #     print('  + preditions: %s 个xfc.' % len(y), [yy.max(1) for yy in y])
        return toc

if __name__ == '__main__':
    inputs = torch.randn(4, 3, 224, 224)
    for key,value in annotations_new.items():
        net = value
        calculate_params_scale(net, name=key, format='million')
        calculate_FLOPs_scale(net, inputs, use_gpu=False, multiply_adds=False)
        calculate_layers_num(net)
        sum_time = 0
        for i in range(20):
            sum_time += calculate_time_cost(net, inputs, use_gpu=False, toc=1, pritout=False)
        print('  + time cost: %.2f ms\t' % (sum_time/20))
