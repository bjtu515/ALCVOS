import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import math
import operator
from  scipy import  misc

#Optimization algorithm for discrete point cluster

def setZero(mask,mark):
    [high,width] = np.shape(mask)
    for i in range(high):
        for j in range(width):
            if(mask[i][j]==mark):
                mask[i][j]=0

def toMask(image):
    [h,w] = np.shape(image)
    for i in range(h):
        for j in range(w):
            if(image[i][j]!=0):
                image[i][j]=255
    return image


def distance(coor1,coor2):
    return math.sqrt(pow(coor1[0]-coor2[0],2)+pow(coor1[1]-coor2[1],2))

def countRegion(mask,union):
    count_each_region = {}
    coordinate = {}
    max_key = [0,0]
    [high,width] = np.shape(mask)
    for i in range(high):
        for j in range(width):
            key = mask[i][j]
            if key!=0:
                while union[key]!=key:
                    key = union[key]
                mask[i][j]=key
                if key in count_each_region:
                    count_each_region[key]=count_each_region[key]+1
                    if count_each_region[key]>max_key[1]:
                        max_key[0] = key
                        max_key[1] = count_each_region[key]
                    coor = coordinate[key]
                    if i<coor[0]:
                        coor[0]=i
                    if i>coor[1]:
                        coor[1]=i
                    if j<coor[2]:
                        coor[2]=j
                    if j>coor[3]:
                        coor[3]=j
                    coordinate[key]=coor
                else:
                    count_each_region[key]=1
                    coor = [i,i,j,j]
                    coordinate[key]=coor
    return count_each_region,max_key,coordinate

def optimiConnecRegion(img,distanc,thresholding):
    [high,width] = np.shape(img)
    mask  = np.zeros_like(img)
    mark  = 0
    union = {}
    for i in range (high):
        for j in range(width):
            if i==0 and j==0:
                if img[i][j]==255:
                    mark=mark+1
                    mask[i][j]=mark
                    union[mark]=mark
            if i==0 and j!=0:
                if img[i][j]==255:
                    left = mask[i][j-1]
                    if left!=0:
                        mask[i][j]=left
                    else:
                        mark = mark +1
                        mask[i][j]=mark
                        union[mark]=mark
            if  j==0 and i!=0:
                if img[i][j]==255:
                    up  = mask[i-1][j]
                    up_right = mask[i-1][j+1]
                    if up==0 and up_right==0:
                        mark = mark+1
                        mask[i][j]=mark
                        union[mark]=mark
                    if up==0 and up_right!=0:
                        mask[i][j]=up_right
                    if up_right==0 and up!=0:
                        mask[i][j]=up
                    if up!=0 and up_right!=0:
                        if up==up_right:
                            mask[i][j]=up
                        else:
                            mi = min(up,up_right)
                            mask[i][j]=mi
                            if up<up_right:
                                union[up_right]=up
                            else:
                                union[up]=up_right
            if i!=0 and j!=0:
                if img[i][j]==255:
                    up = mask[i-1][j]
                    up_left = mask[i-1][j-1]
                    left = mask[i][j-1]
                    up_right = 0
                    if j+1<width:
                        up_right = mask[i-1][j+1]
                    ma = max(max(max(up,up_left),up_right),left)
                    if ma==0:
                        mark = mark+1
                        mask[i][j]=mark
                        union[mark]=mark
                    else:
                        if up==up_right and up_right==up_left and up==left:
                            mask[i][j]=up
                        else:
                            mi = min(min(min(up, up_left), up_right), left)
                            if mi!=0:
                                mask[i][j]=mi
                                if up!=mi:
                                    union[up]=mi
                                if up_right!=mi:
                                    union[up_right]=mi
                                if up_left!=mi:
                                    union[up_left]=mi
                                if left!=mi:
                                    union[left]=mi
                            else:
                                n_zero = []
                                if up!=0:
                                    n_zero.append(up)
                                if up_left!=0:
                                    n_zero.append(up_left)
                                if up_right!=0:
                                    n_zero.append(up_right)
                                if left!=0:
                                    n_zero.append(left)
                                mi1 = min(n_zero)
                                mask[i][j]=mi1
                                for it in n_zero:
                                    if it!=mi1:
                                        union[it]=mi1
    # count = 0
    # ks = dict.keys(union)
    # for i in ks:
    #     if union[i]==i:
    #         count=count+1
    count_each_region,max_key,coordinate = countRegion(mask,union)
    keys = dict.keys(count_each_region)
    coor_max = coordinate[max_key[0]]
    center_max = [(coor_max[0]+coor_max[1])/2,(coor_max[2]+coor_max[3])/2]
    for key in keys:
        num = count_each_region[key]
        if float(num)/max_key[1] < thresholding:
            setZero(mask,key)
        else:
            coor1 = coordinate[key]
            center = [(coor1[0]+coor1[1])/2,(coor1[2]+coor1[3])/2]
            dis = distance(center_max,center)
            if dis > distanc:
                setZero(mask,key)
    return Image.fromarray(toMask(mask))


# dataset_name = "SEG"
# seq_names =["dance-twirl","drift-straight","goat","horsejump-high","kite-surf","motocross-jump","paragliding-launch","parkour","soapbox","scooter-black"]

# seq_names3 = ['vase-2','cool-car-red-2','salt','twirl-2','bottle-3','minion','wallet','lindor','foot-bowl',
#               'Mey-1','orange_can','coffee','mannequin-2','azrieli-1','charger','toy-2','stapler',
#               'bag','matroshka','piggy','wooden-bowl','candle','lamp-2','nutella']

# seq_names1=["horsejump-high"]
# seq_names_seg = ['bird','birdfall','bmx','drift','penguin','frog','hummingbird','monkey','soldier','parachute','monkeydog']

# for seq_name in seq_names_seg:
#      origi_frames = sorted(os.listdir(os.path.join(os.path.join(dataset_name, 'Results','Segmentations','480p','seg', seq_name))))
#      result_path = os.path.join(os.path.join(dataset_name, 'Results','Segmentations','480p','seg-op', seq_name))
#      i=0
#      for img_p in origi_frames:
#         print(img_p)
#         frame_num = img_p.split('.')[0]
#         mask = np.array(Image.open(os.path.join(os.path.join(dataset_name, 'Results','Segmentations','480p','seg', seq_name, img_p))).convert('L'))
#         mask = optimiConnecRegion(mask,100,0.2)
#         misc.imsave(os.path.join(result_path,img_p),mask)

