import cv2
import os
import pandas as pd
import numpy as np
import math
import keras
from PIL import Image, ImageDraw
import face_recognition
import matplotlib.pyplot as plt
#from os.path import basename
#import pathlib
#from pathlib import Path
#import os
#import random
#from PIL import Image, ImageDraw,ImageFont
#from tqdm import tqdm

'''
This script takes an arbitrary image and aligns, segmentates, and fuses it with a new hairstyle
'''
# neccessary functions
# DISTANCE
def distance(p1,p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.sqrt(dx*dx+dy*dy)

# ROTATE
def scale_rotate_translate(image, angle, center = None, new_center = None, scale = None, resample=Image.BICUBIC):
    if (scale is None) and (center is None):
        return image.rotate(angle=angle, resample=resample)
    nx,ny = x,y = center
    sx=sy=1.0
    if new_center:
        (nx,ny) = new_center
    if scale:
        (sx,sy) = (scale, scale)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    a = cosine/sx
    b = sine/sx
    c = x-nx*a-ny*b
    d = -sine/sy
    e = cosine/sy
    f = y-nx*d-ny*e
    return image.transform(image.size, Image.AFFINE, (a,b,c,d,e,f), resample=resample)

# CROP
def crop_face(image, eye_left=(0,0), eye_right=(0,0), offset_pct=(0.3,0.3), dest_sz = (600,600)):
    # calculate offsets in original image
    offset_h = math.floor(float(offset_pct[0])*dest_sz[0])
    offset_v = math.floor(float(offset_pct[1])*dest_sz[1])
    # get the direction
    eye_direction = (eye_right[0] - eye_left[0], eye_right[1] - eye_left[1])
    # calc rotation angle in radians
    rotation = -math.atan2(float(eye_direction[1]),float(eye_direction[0]))
    #print(rotation)
    # distance between them
    dist = distance(eye_left, eye_right)
    # calculate the reference eye-width
    reference = dest_sz[0] - 2.0*offset_h
    # scale factor
    scale = float(dist)/float(reference)
    # rotate original around the left eye

    image = scale_rotate_translate(image, center=eye_left, angle=rotation)
    # crop the rotated image
    crop_xy = (eye_left[0] - scale*offset_h, eye_left[1] - scale*offset_v)
    crop_size = (dest_sz[0]*scale, dest_sz[1]*scale)
    image = image.crop((int(crop_xy[0]), int(crop_xy[1]), int(crop_xy[0]+crop_size[0]), int(crop_xy[1]+crop_size[1])))
    # resize it
    image = image.resize(dest_sz, Image.ANTIALIAS)
    return image


# MAKE FACE
def make_face_df(image_select,filenum):
    # This function looks at one image, draws points and saves points to DF
    pts = []
   # filenum = 0   # need this to iterate through the dataframe to append rows
    face = 0
    image = face_recognition.load_image_file(image_select)
    face_landmarks_list = face_recognition.face_landmarks(image)
    
    for face_landmarks in face_landmarks_list:
        face += 1
        if face >1:    # this will only measure one face per image
            break
        else:
            # Print the location of each facial feature in this image
            facial_features = [
                'chin',
                'left_eyebrow',
                'right_eyebrow',
                'nose_bridge',
                'nose_tip',
                'left_eye',
                'right_eye',
                'top_lip',
                'bottom_lip'
                ]

            for facial_feature in facial_features:
                # put each point in a COLUMN
                for  point in  face_landmarks[facial_feature]:
                    for pix in point:
                        pts.append(pix)
               
        pil_image = Image.fromarray(image)
        d = ImageDraw.Draw(pil_image)
        
        eyes = []
        lex = pts[72]
        ley = pts[73]
        rex = pts[90]
        rey = pts[91]
        eyes.append(pts[72:74])
        eyes.append(pts[90:92])

        image =  Image.open(image_select)
        crop_image = crop_face(image, eye_left=(lex, ley), eye_right=(rex, rey), offset_pct=(0.35,0.6), dest_sz=(300,300))
        try:
            crop_image.save(str(image_select)+"_NEW_cropped.jpg")
        except:
            continue
        # crop_image.show()
        
        nn = str(image_select)+"_NEW_cropped.jpg"
        pts = []
        face = 0
        image = face_recognition.load_image_file(nn)
        face_landmarks_list = face_recognition.face_landmarks(image)

        for face_landmarks in face_landmarks_list:
            face += 1
            if face >1:    # this will only measure one face per image
                break
            else:
                # Print the location of each facial feature in this image
                facial_features2 = [
                    'chin',
                    'left_eyebrow',
                    'right_eyebrow',
                    'nose_bridge',
                    'nose_tip',
                    'left_eye',
                    'right_eye',
                    'top_lip',
                    'bottom_lip'
                    ]

                for facial_feature in facial_features2:
                    # put each point in a COLUMN
                    for  point in  face_landmarks[facial_feature]:
                        for pix in point:
                            pts.append(pix)

            i = 0
            for j in range(0,17):
                if i != 16:
                    if i != 17:
                        px = pts[i]
                        py = pts[i+1]
                        chin_x = pts[16]   # always the chin x
                        chin_y = pts[17]   # always the chin y

                        x_diff = float(px - chin_x)

                        if(py == chin_y):
                            y_diff = 0.1
                        if(py < chin_y):
                            y_diff = float(np.absolute(py-chin_y))
                        if(py > chin_y):
                            y_diff = 0.1
                            print("Error: facial feature is located below the chin.")

                        angle = np.absolute(math.degrees(math.atan(x_diff/y_diff)))

                        pts.append(angle)
                i += 2
        
            pil_image = Image.fromarray(image)
            d = ImageDraw.Draw(pil_image)

            for facial_feature in facial_features2:
                    #d.line(face_landmarks[facial_feature], width=5)
                    d.point(face_landmarks[facial_feature], fill = (255,255,255))
            
            pil_image.save(str(image_select) + '_NEW_rotated_pts.jpg', 'JPEG', quality = 100)
            
            # take_measurements width & height measurements
        msmt = []
        a = pts[0]   ## point 1 x - left side of face
        b = pts[1]   ## point 1 y
        c = pts[32]  ## point 17 x - right side of face
        d = pts[33]  ## point 17 y

        e = pts[16]  ## point 9 x - chin
        f = pts[17]  ## point 9 y - chin
        #Visual inspection indicates that point 29 is the middle of the face,
        #so the height of the face is 2X the height between chin & point 29 which are coordinates 56 and 57
        g = pts[56]  # point 29's x coordinate (mid-face point)
        h = pts[57]   # point 29's y coordinate
        
        i = pts[12]    # point 7 x   for jaw width
        j = pts[13]    # point 7 y   for jaw width
        k = pts[20]    # point 11 x  for jaw width
        l = pts[21]    # point 11 y  for jaw width
             
        m = pts[8]     # point 5 x   for mid jaw width
        n = pts[9]     # point 5 y   for mid jaw width
        o = pts[24]    # point 13 x   for mid jaw width
        p = pts[25]    # point 13 y   for mid jaw width


        face_width = np.sqrt(np.square(a - c) + np.square(b - d))
        pts.append(face_width)
        face_height = np.sqrt(np.square(e - g) + np.square(f - h)) * 2   # double the height to the mid-point
        pts.append(face_height)
        height_to_width = face_height/face_width
        
        pts.append(height_to_width)
        
        # JAW width (7-11)
        jaw_width = np.sqrt(np.square(i-k) + np.square(j-l))
        pts.append(jaw_width)
        jaw_width_to_face_width =  jaw_width/face_width
        pts.append(jaw_width_to_face_width)
        
        # mid-JAW width (5-13)
        mid_jaw_width = np.sqrt(np.square(m-o) + np.square(n-p))
        pts.append(mid_jaw_width)
        mid_jaw_width_to_jaw_width =  mid_jaw_width/jaw_width
        pts.append(mid_jaw_width_to_jaw_width)
        
        
        
#Get the uploaded image

path = 'prova2.jpeg' #path to uploaded image
make_face_df(path,1)
path_aligned= path + '_NEW_cropped.jpg'
image = cv2.imread(path_aligned,0)
image_color = cv2.imread(path_aligned)

#Load Model
model = keras.models.load_model('2KImages_unet_neck_128')

#resize image(gray and colored) to input size of model
img = []
image_resize = cv2.resize(image, (128,128))
img.append(image_resize)
img = np.array(img)

img_color = []
image_color_resize = cv2.resize(image_color, (512,512))
img_color.append(image_color_resize)
img_color = np.array(img_color)


# get mask of face
y_pred_face=model.predict(img)
cv2.imwrite('mask_face_segment.jpg', y_pred_face[0])
mask_face = cv2.imread('mask_face_segment.jpg',0)
# resize it to colored image
mask_face = cv2.resize(mask_face, (512,512))

# apply mask to get colored face
image_face = cv2.bitwise_and(img_color[0],img_color[0],mask = mask_face)

# get hairstyle image and mask

hair_mask = cv2.imread('mask3.jpg',0)
#resize it to colored image
hair_mask = cv2.resize(hair_mask, (512,512))
#triple mask for colored image
hair_mask = np.repeat(hair_mask[:, :, np.newaxis], 3, axis=2)


hair = cv2.imread('hairstyle3.jpg')
hair = cv2.resize(hair, (512,512))

# apply inverse hairstyle mask on colored image

remove_hairmask = np.where(hair_mask == 0, image_face, 0)

#add hairstyle
finalresult = cv2.add(hair, remove_hairmask)
cv2.imwrite('finalfusion.jpg',finalresult)
finalresult = cv2.cvtColor(finalresult, cv2.COLOR_RGB2BGR)

fig, axs = plt.subplots(3, 3,figsize=(12,12))
original = cv2.imread(path)
original = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
image_color = cv2.cvtColor(image_color, cv2.COLOR_RGB2BGR)
image_face = cv2.cvtColor(image_face, cv2.COLOR_RGB2BGR)
remove_hairmask = cv2.cvtColor(remove_hairmask, cv2.COLOR_RGB2BGR)
hair = cv2.cvtColor(hair, cv2.COLOR_RGB2BGR)

axs[0,0].imshow(original)
axs[0,1].imshow(image,cmap="gray")
axs[0,2].imshow(image_color)
axs[1,0].imshow(mask_face)
axs[1,1].imshow(image_face)
axs[1,2].imshow(hair)
axs[2,0].imshow(remove_hairmask)
axs[2,1].imshow(finalresult)
plt.show()
#axs[2,2].imshow(y_pred[0])


