import io
import re
import os
import cv2
import flask
import numpy as np
from PIL import Image
from tabledef import *
from datetime import timedelta
from flask import request, session, jsonify, redirect
from matplotlib import pyplot as plt
from sqlalchemy.orm import sessionmaker
import face_recognition
import imutils
import shutil
import base64
from flask_cors import CORS
import msgspec

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#-------Connect to Database------#
engine = create_engine('sqlite:///login_db.db', echo=True)

#-------Import our model from folder-------#
from anti_spoofing.face_anti_spoofing import detect
from id_card_recognition.sift_flann import sift, match
from id_card_recognition.utils import findFaces
# from face_rrecognition.encode_faces import encode_face
# from face_rrecognition.recognize_faces import recognize

application = flask.Flask(__name__)
application.secret_key = 'web_app_for_face_recognition_and_liveness' # something super secret

CORS(application)

flg = 0
questionA = ""
image_array = []
number_question = 0
img_temple = []

#------Password Validate------#
def password_check(passwd):
    if len(passwd) < 6:
        return 'la longitud debe ser de al menos 6'
         
    if len(passwd) > 20:
        return 'la longitud no debe ser superior a 20'
         
    if not any(char.isdigit() for char in passwd):
        return 'La contraseña debe tener al menos un número'
         
    if not any(char.isupper() for char in passwd):
        return 'La contraseña debe tener al menos una letra mayúscula'
         
    if not any(char.islower() for char in passwd):
        return 'La contraseña debe tener al menos una letra minúscula'
         
    return 'success'

#---------Recognize user's face----------
def recognize(img):

    Session = sessionmaker(bind=engine)
    s = Session()
    results = s.query(User).all()

    frame = img    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = imutils.resize(rgb, width=750) # scale down for faster process
    print('[INFO] recognizing faces...')
    boxes = face_recognition.face_locations(rgb, model='hog')
    encodings = face_recognition.face_encodings(rgb, boxes)

    res = "Unknown"
    minn = 1
    for result in results:
        if result.encoded_face is None:
            continue        
        data = result.encoded_face
        dist = face_recognition.face_distance(encodings, data['encodings'][0])[0]
        # loop over the encoded faces
        for encoding in encodings:
            matches = face_recognition.compare_faces(data['encodings'], encoding)
            name = 'Unknown'
            
            if True in matches:
                matchedIdxs = [i for i, b in enumerate(matches) if b]
                counts = {}
                
                for i in matchedIdxs:
                    name = data['names'][i]
                    counts[name] = counts.get(name, 0) + 1
                name = max(counts, key=counts.get)

        if(name != 'Unknown'):
            if(dist < minn):
                minn = dist
                res = name

    return res

#---------Encode user's face----------
def encode_face(img, name, email, detection_method):
    knownEncodings = list()
    knownNames = list()

    image = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    boxes = face_recognition.face_locations(rgb, model = detection_method)
    encodings = face_recognition.face_encodings(rgb, boxes)
    
    for encoding in encodings:
        # add each encoding and name to the list
        knownEncodings.append(encoding)
        knownNames.append(name)

    data = {'encodings': knownEncodings, 'names': knownNames}    
    Session = sessionmaker(bind=engine)
    s = Session()
    user = User(name, email, data)
    s.add(user)
    s.commit()
    
#------Timing out the login session------#

@application.before_request
def make_session_permanent():
    session.permanent = True
    application.permanent_session_lifetime = timedelta(minutes=8)

#------Index------#
@application.route('/')
def index():
    return flask.render_template("index.html")

#------Sign Up-------#
@application.route('/sign_up')
def sign_up():
    return flask.render_template("sign_up.html")

#------IDCard Verification-------#
@application.route('/id_card')
def id_card():
    if not session.get('logged_in'):
        return sign_up()
    else:
        return flask.render_template("id_card.html")
        
@application.route('/id_verification', methods = ["POST"])
def id_verification():

    data = {'sucess': False, 'face': True}
    if flask.request.method == "POST":
        
        direction = str(request.form['direction'])
        if flask.request.files.get("image"):
        
            name = session['username']
            img = flask.request.files["image"].read()
            img = np.array(Image.open(io.BytesIO(img)))
            dirName = "static/dataset/" + name
            if not os.path.exists(dirName):
                os.makedirs(dirName)
                print("Directory " , dirName ,  " Created ")

            if direction == 'front':
                plt.imsave("static/dataset/" + name + "/front.png", cv2.resize(img, (100, 170)))
                text, face = sift(img)      
                print(text)
                if text == 'Not_ID_Card':
                    data['success'] = False
                elif face is None:
                    data['success'] = True
                    data['face'] = False
                else:
                    session['id_capture'] = True
                    data['success'] = True
                    plt.imsave("static/dataset/" + name + "/crop_face.png", face)
            elif direction == 'back':
                plt.imsave("static/dataset/" + name + "/back.png", img)

    return flask.jsonify(data)

#------Liveness And Face Recognition-------#
@application.route('/signature')
def signature():
    if not session.get('id_capture'):
        return id_card()
    else:
        return flask.render_template("signature.html")
    # return flask.render_template("signature.html")

def face_recog(name):
    known_image = face_recognition.load_image_file("static/dataset/" + name + "/crop_face.png")
    known_image_encoding = face_recognition.face_encodings(known_image)[0]
    unknown_image = face_recognition.load_image_file("static/dataset/" + name + "/liveness_face.png")
    face_encodings = face_recognition.face_encodings(unknown_image)
    face_distance = face_recognition.face_distance(face_encodings, known_image_encoding)[0]
    print(face_distance)
    
    if face_distance < 0.6:
        return "pass"
    else:
        return "fail"


flg = 0

@application.route("/predict", methods=["POST"])
def predict():
    
    data = {'success': False,
            'id_ver': False,
            'is_cal': False,
            'is_start': False,
            'final': False,
            'not_find_face':False,
            'token': "",
            'name': "",
            'ide': 0}

    if flask.request.method == "POST":
       
        question = str(request.form['question'])
        latitude = str(request.form['latitude'])
        longitude = str(request.form['longitude'])
        global flg
        global number_question
        global image_array
        global questionA
        
        session['latitude'] = latitude
        session['longitude'] = longitude

        name = session['username']
        email = session['useremail']
        if flask.request.files.get("image"):
          
            img = flask.request.files["image"].read()
            img = np.array(Image.open(io.BytesIO(img)))

            flag = 0
            challenge_res = 'fail'
            if question == 'final_img':
                if flg == 0:
                    len, img_find_face = sift(img)
                    data['final'] = True
                    flg = 1
                    if not img_find_face is None:
                        plt.imsave("static/dataset/" + name + "/liveness_face1.png", cv2.resize(img, (100, 100)))
                        plt.imsave("static/dataset/" + name + "/liveness_face.png", img)
                        result = face_recog(name)
                        ide = session['useride']
                        token = session['usertoken']
                        encode_face(img, name, email, 'hog')
                        print("face_recognition", result)
                        if(result == "pass"):
                            data['id_ver'] = True
                            data['ide'] = ide
                            data['token'] = token
                            data["name"] = name
                        else :
                            session['id_capture'] = False
                    else: data['not_find_face'] = True
                    return flask.jsonify(data)
            elif number_question < 25:
                flg = 0
                if number_question == 1:
                    questionA = question
                number_question += 1
                image_array.append(img)
            if number_question == 25:
                flag = 1
                flg = 0
                number_question = 0
                challenge_res = detect(image_array, questionA)
                image_array = []

            if flag == 1:
                data['is_cal'] = True
                if challenge_res == 'pass':
                    data['success'] = True

    return flask.jsonify(data)

@application.route('/f_recognition')
def f_recognition():
    if not session.get('logged_in'):
        return flask.render_template("sign_up.html")
    return flask.render_template("face_recognition.html")

@application.route('/recognition', methods=["POST"])
def recognition():
    data = {'success': False,
            'is_cal': False,
            'is_start': False,
            'final': False,
            'name': "Unknown",
            'not_find_face': False}

    if flask.request.method == "POST":
       
        question = str(request.form['question'])
        latitude = str(request.form['latitude'])
        longitude = str(request.form['longitude'])
        session['latitude'] = latitude
        session['longitude'] = longitude
        global flg
        global number_question
        global image_array
        global questionA
        
        if flask.request.files.get("image"):
          
            img = flask.request.files["image"].read()
            img = np.array(Image.open(io.BytesIO(img)))

            flag = 0
            if question == 'final_img':
                if flg ==0:
                    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_find_face = findFaces(img1)
                    data['final'] = True
                    flg = 1
                    # print(img_find_face)
                    if not img_find_face is None:
                        name = recognize(img)
                        if name != 'Unknown':
                            data['name'] = name
                        else :
                            session['id_capture'] = False
                    else: data['not_find_face'] = True
                    return flask.jsonify(data)
            if number_question < 25:
                flg = 0
                if number_question == 1:
                    questionA = question
                number_question += 1
                image_array.append(img)
            challenge_res = 'fail'
            if number_question == 25:
                flag = 1
                number_question = 0
                challenge_res = detect(image_array, questionA)
                image_array = []

            if flag == 1:
                data['is_cal'] = True
                if challenge_res == 'pass':
                    data['success'] = True
    return flask.jsonify(data)

#------------api---------------

@application.route('/redirector', methods=["POST"])
def redirector():
    if flask.request.method == "POST":
        status = str(request.form["status"])
        ide = int(request.form["ide"])
        print(status, ide)
        if ide > 0:
            url = str(request.form["url"])
            name = str(request.form["name"])
            email = str(request.form["email"])
            token = str(request.form["token"])
            ide = int(request.form['ide'])
            session['logged_in'] = True
            session['username'] = name
            session['useremail'] = email
            session['usertoken'] = token
            session['useride'] = ide
            session['url'] = url
            if status == "OK":
                return f_recognition()
            else: return id_card()
   
#----------clear_pic-----------#
@application.route('/clear_pic')
def clear_pic():
    url = session['url']
    name = session['username']
    token = session['usertoken']
    latitude = session['latitude']
    longitude = session['longitude']
    url = url + '/?token=' + token + '&gpslatitude=' + latitude + '&gpslongitude=' + longitude 
    if os.path.isdir('static/dataset/' + name):
        shutil.rmtree('static/dataset/' + name)
    session['logged_in'] = False
    session['id_capture'] = False
    session['username'] = ''
    session['useremail'] = ''
    session['usertoken'] = ''
    session['latitude'] = ''
    session['longitude'] = ''
    session['url'] = ''
    session['useride'] = 0
    return redirect(url)


#-----------delete_all------------
@application.route('/delete_all')
def delete_all():
    Session = sessionmaker(bind=engine)
    s = Session()
    s.query(User).delete()
    s.commit()
    session['logged_in'] = False
    session['id_capture'] = False
    session['username'] = ''
    session['useremail'] = ''
    session['usertoken'] = ''
    session['useride'] = 0
    return index()

if __name__ == "__main__":

    print("** Starting Flask server.........Please wait until the server starts ")
    print('Loading the Neural Network......\n')

    application.run(host = '0.0.0.0', port = '8080')
    
  
