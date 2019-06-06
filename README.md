# ALCVOS
Adaptive Convolutional Neural Network for Large Change in Video Object Segmentation



Paper:
Adaptive Convolutional Neural Network for Large Change in Video Object Segmentation
Hui Yin, Lin Yang, Hongli Xu, and Jin Wan
The Institution of Engineering and Technology(IET)
This is the authors'code described in the above paper. Please cite our paper if you find it useful for your research.

@inproceedings{IET,
  author = {Hui Yin and Lin Yang and Hongli Xu and Jin Wan},
  booktitle = {The Institution of Engineering and Technology(IET)},
  title = {Adaptive Convolutional Neural Network for Large Change in Video Object Segmentation},
  year = {2018}
} 

Install: 
-python 2.7
-tensorflow_gpu==1.1.0 or higher
-other python dependencies:
scipy==0.18.1,matplotlib==1.5.3,numpy==1.12.1,Pillow==4.1.1



Training:
parent model is under models/
To train the model,edit the path and Run python adaptation2.py.


Training the parent network (optional):
All the training sequences of DAVIS 2016 are required to train the parent model
Download the VGG model (55 MB) pretrained on ImageNet, and run python abvos_parent_demo.py to get parent network.

Evaluation:
 To evaluate on DAVIS,you need to download the evaluation code(https://davischallenge.org/davis2016/code.html)