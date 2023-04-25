import cv2 as video
import numpy as np
import cv2

# initializing the video capture
stream = video.VideoCapture(0)

# check if the camera opened successfully or not
if not stream.isOpened():
    print("Error while opening the camera")
    exit()

face_cascade = video.CascadeClassifier(video.data.haarcascades + "haarcascade_frontalface_default.xml")

video.namedWindow("Face Capture")

# define the oval region
center_coordinates = (320, 240)
axes_length = (150, 200) # swapped the axes length to make it vertical
angle = 0
start_angle = 0
end_angle = 360
oval_color = (0, 0, 255)
oval_thickness = 2
oval_color_mask = np.zeros((480, 640, 3), dtype=np.uint8)

# event loop through frames and display video
while True:
    ret, frame = stream.read() # read a frame from the video capture
    height, width = frame.shape[ : 2] # this is the height and width of the canvas
    center = (int(width/3), int(height/2))

    # check if frame is ready successfully or not
    if not ret:
        print("Error in reading ther frame")
        break

    # flipping the frame horizontally
    flipped_frame = video.flip(frame, 1)

    # convert frame to grayscale
    gray_frame = video.cvtColor(flipped_frame, video.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    oval_color_mask = np.zeros((480, 640, 3), dtype=np.uint8)
    oval_color_mask = cv2.ellipse(oval_color_mask, center_coordinates, axes_length, angle, start_angle, end_angle, oval_color, oval_thickness)


    # check if face is within the oval region
    for (x, y, w, h) in faces:
        distance_from_center = np.sqrt((center_coordinates[0] - (x + w / 2)) ** 2 + (center_coordinates[1] - (y + h / 2)) ** 2)

        if distance_from_center < axes_length[0] / 2:
            # set oval color mask to blue
            oval_color_mask = np.zeros((480, 640, 3), dtype=np.uint8)
            oval_color_mask = cv2.ellipse(oval_color_mask, center_coordinates, axes_length, angle, start_angle, end_angle, (0, 255, 0), oval_thickness)

        else:
            # set oval color mask to red
            oval_color_mask = np.zeros((480, 640, 3), dtype=np.uint8)
            oval_color_mask = cv2.ellipse(oval_color_mask, center_coordinates, axes_length, angle, start_angle, end_angle, oval_color, oval_thickness)


    # add oval color mask to the flipped frame
    flipped_frame = cv2.add(flipped_frame, oval_color_mask)

    # adding text to the bottom of the screen
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "Align your face"
    org = (int(width/2) - 100, height-20)
    fontScale = 1
    color = (255, 255, 255)
    thickness = 2
    cv2.putText(flipped_frame, text, org, font, fontScale, color, thickness, cv2.LINE_AA)

    # displaying the flipped frame with oval region
    video.imshow('Face Capture', flipped_frame)


    # check for key presses
    key = video.waitKey(1)
    if key == ord('q') or key == ord('Q'):  # this is to exit the camera output if q is pressed
        break

    if key == ord('s') or key == ord('S'):  # this is to store the image
        video.imwrite('face_profile.jpg', flipped_frame)
        print("Picture taken!")

# release the video capture object and close window
stream.release()
video.destroyAllWindows()