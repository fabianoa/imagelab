import face_recognition
import os
import imutils
import dlib
import cv2

# Now let's use the detector as you would in a normal application.  First we
# will load it from disk.
#detector = dlib.simple_object_detector("detector.svm")
detector = dlib.get_frontal_face_detector()
#predictor = dlib.shape_predictor(args["shape_predictor"])

# make a list of all the available images
images = os.listdir('../images')
print(images)

# Video capture source
cap = cv2.VideoCapture(0)

# We can look at the HOG filter we learned.  It should look like a face.  Neat!
win_det = dlib.image_window()
win_det.set_image(detector)

win = dlib.image_window()

while True:

    ret, image = cap.read()
    image = imutils.resize(image, width=800)

    rects = detector(image)
     
    image_to_be_matched_encoded = face_recognition.face_encodings(
    image)
    
    if(len(image_to_be_matched_encoded)>0):
        image_to_be_matched_encoded = image_to_be_matched_encoded[0]

        for img in images:
            # load the image
            current_image = face_recognition.load_image_file("../images/" + img)
            # encode the loaded image into a feature vector
            current_image_encoded = face_recognition.face_encodings(current_image)[0]
            # match your image with the image and check if it matches
            result = face_recognition.compare_faces([image_to_be_matched_encoded], current_image_encoded)
            # check if it was a match
            if result[0] == True:
                print("Matched: " + img)
            else:
                print("Not matched: " + img)

    #for k, d in enumerate(rects):
        #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        #    k, d.left(), d.top(), d.right(), d.bottom()))
  

    win.clear_overlay()
    win.set_image(image)
    #win.add_overlay(rects)
    win.add_overlay(rects,dlib.rgb_pixel(255,0,0))