import face_recognition
import os
import imutils
import dlib
import cv2

def __draw_label(img, text, pos, bg_color):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    color = (0, 0, 0)
    thickness = cv2.FILLED
    margin = 2

    txt_size = cv2.getTextSize(text, font_face, scale, thickness)

    end_x = pos[0] + txt_size[0][0] + margin
    end_y = pos[1] - txt_size[0][1] - margin

    cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
    cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)

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

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

# We can look at the HOG filter we learned.  It should look like a face.  Neat!
win_det = dlib.image_window()
win_det.set_image(detector)
win = dlib.image_window()

# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    
    image = imutils.resize(frame, width=800)
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
                __draw_label(frame, img, (20,20), (255,0,0))
            
    # Display the resulting frame
    cv2.imshow('Frame',frame)

    # Press Q on keyboard to  exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  # Break the loop
  else: 
    break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()









