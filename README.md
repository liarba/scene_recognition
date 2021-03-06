# DF<sup>2</sup>Net: A Discriminative Feature Learning and Fusion Network for RGB-D Indoor Scene Classification
This is the code for the paper [DF<sup>2</sup>Net: A Discriminative Feature Learning and Fusion Network for RGB-D Indoor Scene Classification](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/download/16730/16293)(AAAI 2018).
![](https://github.com/liarba/scene_recognition/blob/master/framework_image/0001.jpg)
## Installation
1. Install prerequsites for [caffe](http://caffe.berkeleyvision.org/installation.html). The caffe version we use can be found in our `caffe_dev` repo.
2. Add the loss layers to the `caffe_dev` source and include path. 
3. Compile the `caffe_dev` Github submodule
## Our Trained Models
You can download our trained model on SUN RGB-D Dataset and NYU Dataset V2 at [Baidu Netdisk](https://pan.baidu.com/s/1tz8gFuY40bhujQtE-5fb5Q) or [Google Driver](https://drive.google.com/drive/folders/11H79eEfH9AgbuMmu3_z3pX631M9NGfOp?usp=sharing).
## Our Converted HHA Images
We use the Saurabh Gupta's [code](https://github.com/s-gupta/rcnn-depth/blob/master/rcnn/saveHHA.m) to convert the depth images.
We also release our converted HHA images of [SUN RGB-D Dataset](https://drive.google.com/open?id=1TvFiLwhEmTxn8r7XNE2bpVBxoqDqLYsf) and [NYU Depth V2 Dataset](https://drive.google.com/open?id=14nZI0wCu-Ki5AKHKit1dcqpBO1rN2H9p) at Google Driver.
