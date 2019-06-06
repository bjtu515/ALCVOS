from PIL import Image as Image
import matplotlib.pyplot as plt
from scipy import spatial
import numpy as  np

#to get Dappreance

def dist(im1,mk1,im2,mk2):
   [h,w]=np.shape(mk1)
   for i in range(h):
      for j in range(w):
         if(mk1[i,j]==0):
            im1[i,j,:]=0
         if(mk2[i,j]==0):
            im2[i,j,:]=0
   h1=np.histogram(im1,bins=np.arange(64))
   h2=np.histogram(im2,bins=np.arange(64))
#calcute cosine distance
   return spatial.distance.cosine(h1[0][1:64], h2[0][1:64])   





# im1=np.array(Image.open('00001.jpg'))
# mk1=np.array(Image.open('00001.png'))
# im2=np.array(Image.open('00000.jpg'))
# mk2=np.array(Image.open('00000.png'))
#
#
# print(dist(im1,mk1,im2,mk2))


