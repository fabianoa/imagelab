import imutils
import dlib
import cv2

# Now let's use the detector as you would in a normal application.  First we
# will load it from disk.
#detector = dlib.simple_object_detector("detector.svm")
detector = dlib.get_frontal_face_detector()
#predictor = dlib.shape_predictor(args["shape_predictor"])


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

    for k, d in enumerate(rects):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))

    win.clear_overlay()
    win.set_image(image)
    win.add_overlay(rects)