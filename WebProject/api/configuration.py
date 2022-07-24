from decouple import config
class Config:
    pass

class ProductionConfig(Config):
    DEBUG = False
    SQLALCHEMY_DATABASE_URI = config('DATABASE_URL', default='localhost')
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