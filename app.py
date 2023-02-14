import io
import re
import os
import cv2
import flask
import numpy as np
from PIL import Image
from tabledef import *
from datetime import timedelta
from flask import request, session
from matplotlib import pyplot as plt
from sqlalchemy.orm import sessionmaker
import face_recognition
import imutils

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
                # print(counts)    
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
    query = s.query(User).filter(User.email.in_([email]))
    result = query.first()
    result.encoded_face = data
    s.add(result)
    s.commit()

#------Timing out the login session------#

@application.before_request
def make_session_permanent():
    session.permanent = True
    application.permanent_session_lifetime = timedelta(minutes=8)

#------Index------#
@application.route('/')
def index():
    # if not session.get('logged_in'):
    return flask.render_template("index.html")

#------Dashboard-------#
@application.route('/dashboard')
def dashboard():
    if not session.get('final_login'):
        return index()
    return flask.render_template('dashboard.html')

#------Sign Up-------#
@application.route('/sign_up')
def sign_up():
    return flask.render_template("sign_up.html")

@application.route('/signup_user', methods=['GET', "POST"])
def signup_user():
   
    msg = ''
    engine = create_engine('sqlite:///login_db.db', echo=True)
    if flask.request.method == 'POST' and 'username' in flask.request.form and 'password' in flask.request.form and 'email' in flask.request.form :
   
        username = str(request.form['username'])
        password = str(request.form['password'])
        email = str(request.form['email'])

        Session = sessionmaker(bind=engine)
        s = Session()
        query = s.query(User).filter(User.email.in_([email]))
        result = query.first()
        pass_validate = password_check(password)

        if result:
            msg = 'la cuenta ya existe!'
        elif not re.match(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b', email):
            msg = 'Dirección de correo electrónico no válida!'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'El nombre de usuario debe contener solo caracteres y números!'
        elif pass_validate != 'success':
            msg = pass_validate
        elif not username or not password or not email:
            msg = 'por favor rellena el formulario!'
        else :
            user = User(username, email, password, None)
            s.add(user)
            s.commit()
            session['logged_in'] = True
            session['username'] = username
            session['useremail'] = email
            msg = 'Se ha registrado exitosamente!'
            return id_card()
    
    return flask.render_template('sign_up.html', msg = msg)
   
#------Sign Out-------#
@application.route("/logout", methods=['POST'])
def logout():
    # logging out the user
    session['logged_in'] = False
    session['id_capture'] = False
    session['username'] = ''
    session['useremail'] = ''
    return index()

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
            dirName = "dataset/" + name
            if not os.path.exists(dirName):
                os.makedirs(dirName)
                print("Directory " , dirName ,  " Created ")

            if direction == 'front':
                plt.imsave("dataset/" + name + "/front.png", img)
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
                    plt.imsave("dataset/" + name + "/crop_face.png", face)
            elif direction == 'back':
                plt.imsave("dataset/" + name + "/back.png", img)

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
    known_image = face_recognition.load_image_file("./dataset/" + name + "/crop_face.png")
    known_image_encoding = face_recognition.face_encodings(known_image)[0]
    unknown_image = face_recognition.load_image_file("./dataset/" + name + "/liveness_face.png")
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
            'not_find_face':False}

    if flask.request.method == "POST":
       
        count  = int(request.form['count'])
        question = str(request.form['question'])
        global flg
        global number_question
        global image_array
        global questionA
        
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
                        plt.imsave("dataset/" + name + "/liveness_face.png", img)
                        result = face_recog(name)
                        encode_face(img, name, email, 'hog')
                        print("face_recognition", result)
                        if(result == "pass"):
                            data['id_ver'] = True
                            session['final_login'] = True
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
                            session['logged_in'] = True
                            session['final_login'] = True
                            session['username'] = name
                            data['name'] = name
                        else :
                            session['id_capture'] = False
                            session['final_login'] = False
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

if __name__ == "__main__":

    print("** Starting Flask server.........Please wait until the server starts ")
    print('Loading the Neural Network......\n')

    application.run(host = '0.0.0.0', port = '8080')
    
  
