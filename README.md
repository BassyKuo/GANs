# Simple GANs example code

All codes running in __Python3__.
You have to check the packages first, and use `pip install` to install if you miss them:

* Tensorflow==1.3.0
* Numpy
* Scipy
* urllib
* tarfile
* pickle
* six


## Run
```
python 01-cifar10_DCGAN_slim.py --bs 256 --dlr 0.0002 --glr 0.0002
```

## Results & Models

The training results are saved in the `[result]` folder.
The tensorflow checkpoint models are saved in the `__models__` folder.

## Visualization 

You can use `tensorboard` to see the training progress online.
First, toggle the port:
```
$ tensorboard --logdir=summary/
Tensorboard 0.1.5 at http://ServerName:6006 (Press CTRL+C to quit)
```

Then open your browser, go to `http://localhost:6006`. You can see the histogram of inception score progressing.

## Examples of generated images
* mnist
![mnist] (./sample_data/01-mnist_DCGAN_slim-64bs-0.0002glr-0.0002dlr_9800.jpg "mnist_example")
* cifar10
![cifar10] (./sample_data/01-cifar10_DCGAN_slim-256bs-0.0002glr-0.0002dlr_31600.jpg "cifar10_example")
* imagenet
![imagenet] (./sample_data/01-imagenet-CENTRA_DCGAN_slim-conv2d_BN-normalnoise-256bs-0.0002glr-0.0002dlr_18000.jpg "imagenet_example")
