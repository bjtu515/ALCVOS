import numpy as np
import matplotlib.image as img

#calculate Sm to get Dmotion

def Dist(mask1,mask2):
    [h,w]=np.shape(mask1)
    inters=np.zeros(np.shape(mask1))
    union=np.zeros(np.shape(mask1))
    for i in range(h):
        for j in range(w):
            if(mask1[i,j]==1 and mask2[i,j]==1):
                inters[i,j]=1
            if(mask1[i,j]==1 or mask2[i,j]==1):
                union[i,j]=1
    # print('%d %d',np.count_nonzero(inters),np.count_nonzero(union))
    return np.count_nonzero(inters)/float(np.count_nonzero(union))

#
# im=img.imread('00001.png');
# im1=img.imread('00033.png');
# print(Dist(im,im1))