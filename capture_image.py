import cv2
import mediapipe as mp

# Import the necessary drawing utilities and hands module from the MediaPipe library
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def capture_image(camera):
    """
    This function opens the camera, displays the video feed, and captures an image when the 'c' key is pressed.
    It also uses the MediaPipe library to track the hand while the camera is open.

    Parameters:
    camera (cv2.VideoCapture): The camera object returned by cv2.VideoCapture(0).
    """


    # Initialize the Hands class from the MediaPipe library with a minimum detection confidence and tracking confidence
    with mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:

        while True:
            # Read a frame from the camera
            ret, frame = camera.read()
            if not ret:
                break

            # Convert the frame from BGR (OpenCV's default color format) to RGB (MediaPipe's required color format)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame using the Hands class to detect and track the hand
            results = hands.process(frame)            

            # Convert the frame back to BGR for displaying with OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Wait for a key press and check if it's the 'c' key
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c'):
                # If the 'c' key is pressed, save the current frame as an image named 'captured_image.jpg'
                cv2.imwrite('captured_image.jpg', frame)
                print("Image captured!")
                break

            # If any hands are detected, draw the landmarks on the frame
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Display the frame in a window named 'Camera'
            cv2.imshow('Camera', frame)


    # Release the camera and close all OpenCV windows
    camera.release()
    cv2.destroyAllWindows()


# If this script is run directly (i.e., not imported), open the default camera (index 0) and call the capture_image function
if __name__ == '__main__':
    # sys.stdout = open(os.devnull, 'w')
    camera = cv2.VideoCapture(0)
    capture_image(camera)