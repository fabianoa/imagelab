import face_recognition
import os
import imutils
import dlib
import cv2

def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
 
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

def draw_rects(img, rects,text):
    for (i, rect) in enumerate(rects):
        (x, y, w, h) = rect_to_bb(rect)
        # Draw a box around the face
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, "Face: {}".format(text), (x - 10, y - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        
    
# Now let's use the detector as you would in a normal application.  First we
# will load it from disk.
#detector = dlib.simple_object_detector("detector.svm")
detector = dlib.get_frontal_face_detector()

# make a list of all the available images
images = os.listdir('../images')

# Video capture source
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")


# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:    
    rects = detector(frame)
    image_to_be_matched_encoded = face_recognition.face_encodings(frame)
    
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
                draw_rects(frame, rects,img)
            
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









