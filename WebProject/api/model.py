import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import math
# This script takes one uploaded image and three style images and its corresponding mask
# and fuses them to three new styles

# load hairstyle images
path_style1 = 'image_style1.jpg'
path_style2 = 'image_style2.jpg'
path_style3 = 'image_style3.jpg'

image_style1 = cv2.imread(path_style1)
#image_style1 = cv2.cvtColor(image_style1, cv2.COLOR_RGB2BGR)


image_style1 = cv2.resize(image_style1, (512,512),interpolation = cv2.INTER_AREA)
image_style2 = cv2.imread(path_style2)
#image_style2 = cv2.cvtColor(image_style2, cv2.COLOR_RGB2BGR)


image_style2 = cv2.resize(image_style2, (512,512),interpolation = cv2.INTER_AREA)

image_style3 = cv2.imread(path_style3)
#image_style3 = cv2.cvtColor(image_style3, cv2.COLOR_RGB2BGR)

image_style3 = cv2.resize(image_style3, (512,512),interpolation = cv2.INTER_AREA)

# load mask
path_mask1 = 'mask1.png'
path_mask2 = 'mask2.png'
path_mask3 = 'mask3.png'
mask1 = cv2.imread(path_mask1,0)
mask2 = cv2.imread(path_mask2,0)
mask3 = cv2.imread(path_mask3,0)


# load 'uploaded' image
path_upload = 'upload.jpg'
image_upload = cv2.imread(path_upload)
#image_upload = cv2.cvtColor(image_upload, cv2.COLOR_RGB2BGR)

image_upload = cv2.resize(image_upload, (512,512),interpolation = cv2.INTER_AREA)

# remove everything except skin

path_mask = 'upload_mask.png'
mask_total = cv2.imread(path_mask,0)
mask_upload= np.where(mask_total == 0, mask_total, 255)
face_upload = cv2.bitwise_and(image_upload,image_upload,mask = mask_upload)



# segment style from style image
style1 = cv2.bitwise_and(image_style1,image_style1,mask = mask1)
style2 = cv2.bitwise_and(image_style2,image_style2,mask = mask2)
style3 = cv2.bitwise_and(image_style3,image_style3,mask = mask3)


# apply inverse mask to uploaded' image
upload_sgmt1 = cv2.bitwise_and(face_upload, face_upload, mask=255-mask1)
upload_sgmt2 = cv2.bitwise_and(face_upload, face_upload, mask=255-mask2)
upload_sgmt3 = cv2.bitwise_and(face_upload, face_upload, mask=255-mask3)


# combine the two masked images
result1 = cv2.add(style1, upload_sgmt1)
result2 = cv2.add(style2, upload_sgmt2)
result3 = cv2.add(style3, upload_sgmt3)

# change background to white
result1[result1[:,:,2]==0] = 255
result2[result2[:,:,2]==0] = 255
result3[result3[:,:,2]==0] = 255

cv2.imwrite('fusionstyle1.jpg', result1)
cv2.imwrite('fusionstyle2.jpg', result2)
cv2.imwrite('fusionstyle3.jpg', result3)
