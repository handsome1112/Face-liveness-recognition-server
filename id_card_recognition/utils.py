
import cv2
import numpy as np
from matplotlib import pyplot as plt
# from traceback2 import print_tb
import dlib

detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("id_card_recognition/shape_predictor_68_face_landmarks.dat")

def get_homography_points(M, points):
    new_bb = np.zeros_like(points)

    for i,coord in enumerate(points):

        v = [coord[0],coord[1],1]
        calculated = np.dot(M,v)
        calculated_scaled = calculated/calculated[2]
        new_bb[i] = (calculated_scaled[0], calculated_scaled[1])

    return new_bb


def get_warpPerspective(image_orig, M2, dst):

    img_homog = cv2.warpPerspective(image_orig, M2, (image_orig.shape[1], image_orig.shape[0]))
    dst_h = dst.reshape(4,2)
    new_bb =  get_homography_points(M2, dst_h)
    warp_image = warpImg(img_homog, new_bb , image_orig.shape[0], image_orig.shape[1])

    return warp_image

def get_angle_and_box_coord(dst):
    
    # cv.minAreaRect returns:
    # (center(x, y), (width, height), angle of rotation) = cv2.minAreaRect(c)
    rect = cv2.minAreaRect(dst)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # Retrieve the key parameters of the rotated bounding box
    box_center = (int(rect[0][0]),int(rect[0][1])) 
    box_width = int(rect[1][0])
    box_height = int(rect[1][1])
    angle = int(rect[2])

    if box_width < box_height:
        angle = 90 - angle
    else:
        angle = -angle      
    print("Rotation Angle: " + str(angle) + " degrees")

    return -angle, box

def rotate_bbox(bb, cx, cy, h, w, theta):
    new_bb = np.zeros_like(bb)
    for i,coord in enumerate(bb):
        # opencv calculates standard transformation matrix
        M = cv2.getRotationMatrix2D((cx, cy), theta, 1.0)
        # Grab  the rotation components of the matrix)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cx
        M[1, 2] += (nH / 2) - cy
        # Prepare the vector to be transformed
        v = [coord[0],coord[1],1]
        # Perform the actual rotation and return the image
        calculated = np.dot(M,v)
        new_bb[i] = (calculated[0],calculated[1])
    return new_bb

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # centre
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

def reorder(myPoints):

    myPointsNew = np.zeros_like(myPoints)
    myPoints = myPoints.reshape((4,2))
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]

    diff = np.diff(myPoints, axis = 1)
    
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]

    return myPointsNew


def warpImg(img, points, w, h):

    points = reorder(points)
    pts1 = np.float32(points)
    pts2 = np.float32([[0,0], [w,0], [0,h], [w,h]])
    matrix =  cv2.getPerspectiveTransform(pts1, pts2)
    imgWarp = cv2.warpPerspective(img, matrix, (w,h))

    return imgWarp

def findFaces(image):
    faces = detector(image)
    num_of_faces = len(faces)
    print("Number of Faces:", num_of_faces )
    if (not num_of_faces):
        return None

    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        # print(x1, y1, x2, y2)
        # print(image.left(), image.top(), image.right(), image.bottom())
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255 , 0), 3)
        face_crop = image[y1:y2, x1:x2]
        return face_crop
        

def cropFaceRegions(image,x1, y1, x2, y2):

    face_crop = image[y1:y2, x1:x2]
    #cv2.imshow("crop region:", face_crop)
    plt.imsave("croppedFaces/crop_face.png", face_crop)
    #cv2.waitKey(0)
    return face_crop

def is_two_image_same(img1, img2, face_match_count):
    
    sift = cv2.SIFT_create()
    img1 = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    img2 = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)
    print("Total good matches:", len(good))       
    good = good[:face_match_count]

    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good, None,flags=2)
    plt.title("Face Match")
    plt.imshow(img3, 'gray'),plt.show()
    print("Matches are found - %d/%d" % (len(good), face_match_count))

    if len(good) >= face_match_count:
        print("Faces are similar")
        return True
    else :
        print("Faces are not similar")
    return False

def applyBlur(image):
    return cv2.blur(image,(3,3))

def resizeImage(image):
    h, w = image.shape[0:2]
    return cv2.resize(image, (w+100, h+100), cv2.INTER_LINEAR)