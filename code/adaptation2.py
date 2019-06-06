import os
import sys
import abvos
import time
import tensorflow as tf
from dataset import Dataset
import matplotlib.image as img
import  numpy as np
import distmotion
slim = tf.contrib.slim
import distappar

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
root_folder = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(root_folder))
os.chdir(root_folder)

dataset_name ="SEG"
seq_names =["bmx-trees","breakdance","camel","car-roundabout","car-shadow","cows","dance-twirl","dog","drift-chicane","drift-straight",
            "goat","horsejump-high","kite-surf","libby","motocross-jump","paragliding-launch","parkour","soapbox","scooter-black"]
seq_names2 =["vase-2"]

seq_names1 =["an19","br128","br130","do01013","do01014","do01030","do01055","do02001","m07058","vwc102"]

seq_names3 = ['vase-2','cool-car-red-2','salt','twirl-2','bottle-3','minion','wallet','lindor','foot-bowl',
              'Mey-1','orange_can','coffee','mannequin-2','azrieli-1','charger','toy-2','stapler',
              'bag','matroshka','piggy','wooden-bowl','candle','lamp-2','nutella']

seq_names31 = ['piggy','wooden-bowl','candle','lamp-2','nutella']

seq_names_seg = ['penguin','cheetah','bmx','drift','frog','hummingbird','monkey','soldier']
max_training_iters = 500
update_iters1=10
update_iters2=20
learning_rate = 1e-8
save_step = max_training_iters
side_supervision = 3
display_step = 100
alpha=30
beta=10
delta=0.3

sigma=0.5
n_count=0
parent_path = os.path.join('models', 'ABVOS_parent1', 'ABVOS_parent.ckpt-50000')
start=time.time()
for seq_name in seq_names_seg:
 result_path = os.path.join(dataset_name, 'Results', 'Segmentations', '480p', 'seg',seq_name)
 logs_path = os.path.join('models','seg', seq_name)
 test_frames = sorted(os.listdir(os.path.join(dataset_name, 'JPEGImages', '480p', seq_name)))
 test_imgs = []
 train_imgs = [os.path.join(dataset_name, 'JPEGImages', '480p',seq_name, '00000.jpg') + ' ' +
               os.path.join(dataset_name, 'Annotations', '480p',seq_name, '00000.png')]
 dataset = Dataset(train_imgs, test_imgs, './', data_aug=True)
 t=open('testlistseg', 'r')
 pre=''
 pre1=''
 start=0.00
 mark=0
 flag=0
 i = 1
 zeta = 0
 zeta1= 0
 print('adapdemo2')
 for frame in test_frames:
     n_count=n_count+1
     test_imgs = [os.path.join(dataset_name, 'JPEGImages', '480p' ,seq_name, frame)]
     if i==1:
         with tf.Graph().as_default():
            # print('start training,%d',i)
               global_step = tf.Variable(0, name='global_step', trainable=False)
               abvos.train_finetune(dataset, parent_path, side_supervision, learning_rate, logs_path, max_training_iters,
                                 save_step, display_step, global_step, iter_mean_grad=1, ckpt_name=seq_name)

               i=i+1
     else:
         if i<4:
              with tf.Graph().as_default():
                # print("start segmentation,%d",i)
                checkpoint_path = os.path.join('models','seg', seq_name, seq_name + '.ckpt-' + str(max_training_iters))
                dataset = Dataset(None, test_imgs, './')
                abvos.segmentation(dataset, checkpoint_path, result_path)
                i=i+1
         else:
              line=''
              if(mark==0):
                 mark=1
                 line=t.next()
                 pre = t.next()
              else:
                 line=pre
                 pre=t.next()
              mk1=img.imread(line.split(' ')[1].strip('\n').replace("Annotations/480p/xxxxxx", "Results/Segmentations/480p/seg/"+seq_name))
              mk2=img.imread(pre.split(' ')[1].strip('\n').replace("Annotations/480p/xxxxxx", "Results/Segmentations/480p/seg/"+seq_name))
              im1 = img.imread(line.split(' ')[0].replace("xxxxxx", seq_name))
              im2 = img.imread(pre.split(' ')[0].replace("xxxxxx", seq_name))
              dist_motion= 1 - distmotion.Dist(mk1, mk2)
              dist_appar = distappar.dist(np.array(im1), mk1, np.array(im2), mk2)*100
              dt=dist_motion+dist_appar

              if(dt>delta):


                 train_imgs = [line.replace("xxxxxx", seq_name)]
                 if(flag==0):
                      zeta = int(round(alpha * dist_appar + beta * dist_motion))
                      # with tf.Graph().as_default():
                      #    global_step = tf.Variable(0, name='global_step', trainable=False)
                      #    train_imgs = [os.path.join(dataset_name, 'JPEGImages', '480p', seq_name, '00000.jpg') + ' ' +
                      #                  os.path.join(dataset_name, 'Annotations', '480p', seq_name, '00000.png')]
                      #    dataset = Dataset(train_imgs, [], './')
                      #    checkpoint_path = os.path.join('models', 'adaptdemo2', seq_name,
                      #                                   seq_name + '.ckpt-' + str(max_training_iters))
                      #    abvos.train_refinetune(dataset, checkpoint_path, side_supervision, learning_rate, logs_path,
                      #                           update_iters1, save_step, display_step, global_step, iter_mean_grad=1,
                      #                           ckpt_name=seq_name)
                      with tf.Graph().as_default():
                         global_step = tf.Variable(0, name='global_step', trainable=False)
                         dataset = Dataset(train_imgs, [], './')
                         checkpoint_path = os.path.join('models', 'seg', seq_name,seq_name + '.ckpt-' + str(max_training_iters))
                         abvos.train_refinetune(dataset, checkpoint_path, side_supervision, learning_rate, logs_path,
                                         zeta, save_step, display_step, global_step, iter_mean_grad=1,
                                           ckpt_name=seq_name)
                         flag=1
                 else:
                      zeta1 = int(round(alpha * dist_appar + beta * dist_motion))
                      # with tf.Graph().as_default():
                      #      global_step = tf.Variable(0, name='global_step', trainable=False)
                      #      train_imgs = [os.path.join(dataset_name, 'JPEGImages', '480p', seq_name, '00000.jpg') + ' ' +
                      #              os.path.join(dataset_name, 'Annotations', '480p', seq_name, '00000.png')]
                      #      dataset = Dataset(train_imgs, [], './')
                      #      checkpoint_path = os.path.join('models', 'adaptdemo2', seq_name,seq_name + '.ckpt-' + str(update_iters2))
                      #      abvos.train_refinetune(dataset, checkpoint_path, side_supervision, learning_rate, logs_path,
                      #                       update_iters1, save_step, display_step, global_step, iter_mean_grad=1,
                      #                       ckpt_name=seq_name)
                      with tf.Graph().as_default():
                           global_step = tf.Variable(0, name='global_step', trainable=False)
                           dataset = Dataset(train_imgs, [], './')
                           checkpoint_path = os.path.join('models', 'seg', seq_name,seq_name + '.ckpt-' + str(zeta))
                           abvos.train_refinetune(dataset, checkpoint_path, side_supervision, learning_rate, logs_path,
                                            zeta1, save_step, display_step, global_step, iter_mean_grad=1,
                                            ckpt_name=seq_name)
                      zeta=zeta1
              with tf.Graph().as_default():

                 if(flag==0):
                     checkpoint_path = os.path.join('models', 'seg', seq_name,
                                                    seq_name + '.ckpt-' + str(max_training_iters))
                 else:
                     checkpoint_path = os.path.join('models', 'seg', seq_name,
                                                    seq_name + '.ckpt-' + str(zeta))
                 dataset = Dataset(None, test_imgs, './')
                 abvos.segmentation(dataset, checkpoint_path, result_path)
                 i = i + 1
 t.close()