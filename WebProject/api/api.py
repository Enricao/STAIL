import os
import io
from flask import Flask, request, flash, make_response, render_template, Response
from werkzeug.utils import secure_filename

#from configuration import config
from decouple import config as config_decouple

from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

from PIL import Image as PILmage
import base64

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math


import random
from decouple import config

#--------------------------------------------Config---------------------------------------------------------------------
class Config:
    pass

class ProductionConfig(Config):
    DEBUG = False
    SQLALCHEMY_DATABASE_URI = os.environ['DATABASE_URL'].replace("postgres://", "postgresql://")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    STATIC_FOLDER = '../build'
    STATIC_URL_PATH = '/'
    SECRET_KEY = config('SECRET_KEY', default='localhost')

class DevelopmentConfig(Config):
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = "postgresql://postgres:I2K3E5R9@localhost:5432/stail"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    STATIC_FOLDER = '../public'
    STATIC_URL_PATH = '/'
    SECRET_KEY = "eiwbeivbwpeibvpwejbvqpjbvp"


config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig
}


#-------------------------------------------Initializations-------------------------------------------------------------

application = Flask(__name__,static_folder='../build',static_url_path='/')
environment = config['production']

if config_decouple('PRODUCTION', cast=bool ,default=False):
    print('*******************')
    print('I AM IN PRODUCTION')
    print('*******************')
    environment = config['production']
else:
    print('*******************')
    print('I AM NOT IN PRODUCTION')
    print('*******************')


application.config.from_object(environment)
db = SQLAlchemy(application)
migrate = Migrate(application, db)

#------------------------------------------------Utils------------------------------------------------------------------
def model(update):
    possible = db.session.query(HairStyle.id).all()

    return  [random.choice(possible)[0] for i in range(3)]

def fusion(user_id,style_id):
    user = Image.query.filter_by(id=user_id).first()
    style = HairStyle.query.filter_by(id=style_id).first()
    #print(user.image)
    #print(type(user.image))

    user_img = cv2.imdecode(np.frombuffer(user.image, np.uint8), -1)
    style_img = cv2.imdecode(np.frombuffer(style.image, np.uint8), -1)


    '''
    image_upload = cv2.resize(user_img, (512,512),interpolation = cv2.INTER_AREA)
    image_style1 = cv2.resize(style_img, (512,512),interpolation = cv2.INTER_AREA)


    mask1 = cv2.imread(r'{}.png'.format(int(style_id)-3),0)
    mask_total = cv2.imread(r'upload_mask.png',0)

    #print(mask1.shape)
    #print(mask_total.shape)

    #remove everything except skin
    mask_upload = np.where(mask_total == 0, mask_total, 255)
    anti_mask_upload = np.where(mask_total != 0, mask_total, 255)


    print(mask_upload.shape)


    face_upload = cv2.bitwise_and(image_upload,image_upload,mask = mask_upload)
    background = cv2.bitwise_and(image_upload,image_upload,mask = anti_mask_upload)

    #segment style from style image
    style1 = cv2.bitwise_and(image_style1,image_style1,mask = mask1)

    #apply inverse mask to uploaded image
    upload_sgmt1 = cv2.bitwise_and(face_upload, face_upload, mask=255-mask1)

    #cv2.imwrite('style.png', style1)
    #cv2.imwrite('segment.png',upload_sgmt1)

    #combine the two masked images
    result1 = cv2.add(style1, upload_sgmt1)

    background[result1[:,:,2]!=0] = 0

    ret = cv2.add(result1,background)


    #change background to white
    #result1[result1[:,:,2]==0]
    #background[result[:,:,2]==0] = 0

    #cv2.imwrite('styleFusion.png', ret)

    '''

    return style_img







#-----------------------------------------------Models------------------------------------------------------------------

class Image(db.Model):
    __tablename__ = 'images'

    id = db.Column(db.Integer, primary_key=True)
    image = db.Column(db.LargeBinary)
    name = db.Column(db.String)

    def __init__(self,image,name):
        self.image = image
        self.name = name

    def __repr__(self):
        return '<id {}>'.format(self.id)

class HairStyle(db.Model):
    __tablename__ = 'hairstyles'
    id = db.Column(db.Integer, primary_key=True)
    image = db.Column(db.LargeBinary)
    name = db.Column(db.String)


    def __init__(self, name, image):
        self.image = image
        self.name = name

    def __repr__(self):
        return '<id {}>'.format(self.id)


#------------------------------------------------Requests---------------------------------------------------------------

@application.route('/', defaults={'path': ''})
@application.route('/<path:path>')
def catch_all(path):
    return application.send_static_file("index.html")

@application.route('/api/recommendation', methods=['POST'])
def get_recommendation_from_upload():
    ALLOWED_EXT = {'png', 'jpg', 'jpeg', 'gif'}
    uploads_dir = '/'

    if 'image' not in request.files:
        print('No image part')
        return 400

    image = request.files['image']
    print(image.filename + " received")
    if ('.' not in image.filename or
            image.filename.rsplit('.', 1)[-1].lower() not in ALLOWED_EXT):
        print('Invalid file name')


    filename = secure_filename(image.filename)


    img = Image(image=image.read(),name=filename)
    db.session.add(img)
    db.session.commit()
    recommendation = model(image)
    print(recommendation)

    #relative_path = os.path.join('/','hair1.png')
    return {'image': recommendation,'user': img.id}

@application.route('/api/recommendation/<int:id>',methods=['GET'])
def get_hairstyle_by_id(id):
    img = HairStyle.query.filter_by(id=id).first()
    if not img:
        return 'No image with that id', 404

    #image64 = io.BytesIO(img.image)
    #print(io.BytesIO(img.image))
    #image_n = PILmage.open(image64)
    #print(image_n)
    #print(image_n.show())
    #print(base64.b64encode(img.image))


    return {'hairstyle' : base64.b64encode(img.image).decode('utf-8')}


@application.route('/api/recommendation/user/<int:id>',methods=['GET'])
def get_image_by_id(id):
    img = Image.query.filter_by(id=id).first()
    if not img:
        return 'No image with that id', 404

    return {'user' : base64.b64encode(img.image).decode('utf-8')}

@application.route('/api/recommendation/<string:original>/fuse/<string:hairstyle>',methods=['GET'])
def get_fused_image(original,hairstyle):
    img = fusion(original,hairstyle)
    #fusion_img = Image(image=img,name='fusion')
    #db.session.add(fusion_img)
    #db.session.commit()

    imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = PILmage.fromarray(imageRGB)
    pil_img.save('pil_img.png')
    buff = io.BytesIO()
    pil_img.save(buff, format="PNG")

    #new_image_string = base64.b64encode(buff.getvalue()).decode("utf-8")
    #fusion_ret = Image.query.filter_by(id=fusion_img.id).first()

    return {'fuse' :  base64.b64encode(buff.getvalue()).decode("utf-8")}



if __name__ == '__main__':
    application.run()



#pg_dump --username  postgres --no-owner stail > your-db.dump
#heroku pg:psql --app stail-app-dev < your-db.dump