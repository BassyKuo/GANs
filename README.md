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
python 01-imagenet_DCGAN_slim.py --bs 256 --dlr 0.0002 --glr 0.0002
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
