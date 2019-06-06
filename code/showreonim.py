import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

#write green mask
seq_names =["blackswan","bmx-trees","breakdance","camel","car-roundabout","car-shadow","cows","dance-twirl","dog","drift-chicane","drift-straight",
            "goat","horsejump-high","kite-surf","libby","motocross-jump","paragliding-launch","parkour","soapbox","scooter-black"]

seq_names1=["motorbike"]
for seq_name in seq_names1:
     test_frames = sorted(os.listdir(os.path.join('DAVIS', 'JPEGImages', '480p', seq_name)))
     result_path = os.path.join('DAVIS', 'Results', 'Segmentations', '480p', 'OSVOS', seq_name)
     overlay_color = [0, 255, 0]
     transparency = 0.6
     plt.ion()
     i=0
     for img_p in test_frames:
      frame_num = img_p.split('.')[0]
      img = np.array(Image.open(os.path.join('DAVIS', 'JPEGImages', '480p', seq_name, img_p)))
      mask = np.array(Image.open(os.path.join(result_path, frame_num + '.png')))
      mask = mask/np.max(mask)
      im_over = np.ndarray(img.shape)
      im_over[:, :, 0] = (1 - mask) * img[:, :, 0] + mask * (overlay_color[0]*transparency + (1-transparency)*img[:, :, 0])
      im_over[:, :, 1] = (1 - mask) * img[:, :, 1] + mask * (overlay_color[1]*transparency + (1-transparency)*img[:, :, 1])
      im_over[:, :, 2] = (1 - mask) * img[:, :, 2] + mask * (overlay_color[2]*transparency + (1-transparency)*img[:, :, 2])
      plt.imshow(im_over.astype(np.uint8))
      plt.axis('off')
      plt.show()
      plt.pause(1)
     # print(i)
      i=i+1
      plt.clf()