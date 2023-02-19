from sqlalchemy import *
from sqlalchemy import create_engine, ForeignKey
from sqlalchemy import Column, Date, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref
 
engine = create_engine('sqlite:///login_db.db', echo=True)
Base = declarative_base()
 
########################################################################
class User(Base):
    """"""
    __tablename__ = "users"
 
    id = Column(Integer, primary_key=True)
    username = Column(String)
    email = Column(String)
    encoded_face = Column(PickleType)
 
    #----------------------------------------------------------------------
    def __init__(self, username, email, encoded_face):
        """"""
        self.username = username
        self.email = email
        self.encoded_face = encoded_face
 
# create tables
Base.metadata.create_all(engine)