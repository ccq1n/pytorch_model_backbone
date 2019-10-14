'''ImageNet dataset Path'''
traindir = '/home/user/Downloads/ILSVRC2012/train'
valdir = '/home/user/Downloads/ILSVRC2012/val'

'''Parameter setting'''
input_size = 224  # Input image size
batch_size = 128  # Number of input images
n_worker = 4      # Multiple threads
lr = 0.1          # Learning rate