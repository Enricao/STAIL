from app import db
from sqlalchemy.dialects.postgresql import JSON

list = [87687690]
class Image(db.Model):
    __tablename__ = 'images'

    id = db.Column(db.Integer, primary_key=True)
    image = db.Column(db.Text,unique=True,nullable=False)

    def __init__(self, image):
        self.image = image

    def __repr__(self):
        return '<id {}>'.format(self.id)
