import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt


# %%
def draw_landmarks(image, landmarks, mp_hands, mp_drawing):
    """
    Auxillary function for process_image(). Creates a before and after picture showing hand landmarks drawn onto the original picture.
    """
    
    # Draw the landmarks on the image
    image_with_landmarks = image.copy()
    mp_drawing.draw_landmarks(
        image_with_landmarks,
        landmarks,
        mp_hands.HAND_CONNECTIONS)

    # Display the original image and the image with the detected landmarks using a plot
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    axs[1].imshow(cv2.cvtColor(image_with_landmarks, cv2.COLOR_BGR2RGB))
    axs[1].set_title('Image with Landmarks')
    axs[1].axis('off')
    plt.show()

# %%

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def process_image(image_path, show=False, label=True):
    """
    This function takes in an image, applies the MediaPipe hand model, and stores the landmark x, y, and z positions
    in a list.

    Output list is in the form of all the X coordinates followed by Y and Z [x0, ... ,x20, y0, ... , y20, z0, ... , z20].

    Parameters:
    image_path (str): The path to the image file.
    show (bool): If true, displays the image before and after the MediaPipe model is applied.
    label (bool): If true, output list will be followed up by the label associated with image and the image path.

    Returns:
    list: A list containing the landmark x, y, and z positions.
    """

    # Load the image
    image = cv2.imread(image_path)


    # Initialize the Hands class from the MediaPipe library with a minimum detection confidence and tracking confidence
    with mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:

        # Convert the image from BGR (OpenCV's default color format) to RGB (MediaPipe's required color format)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image using the Hands class to detect and track the hand
        results = hands.process(image_rgb)

        # If any hands are detected, extract the landmarks and draw them on the image
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]

            # Create a list to store the landmark positions
            landmark_positions = []

            # Iterate over the landmarks and append their x, y, and z positions to the list
            for landmark in landmarks.landmark:
                landmark_positions.append([landmark.x, landmark.y, landmark.z])

            # Flattening Data
            landmark_positions = [*np.array(landmark_positions).T.flatten()]

            # Adds a label to for the entry based off of the file it is located in. Also Shows full path.
            if label:
                landmark_positions += [image_path.split("/")[-2]]
                landmark_positions += [image_path]

            # Shows before and after picture showing hand landmarks drawn onto the original picture.
            if show:
                draw_landmarks(image, landmarks, mp_hands, mp_drawing)

            # Converting df into dict

            return landmark_positions

        else:
            # print("\nNo hands detected in the image.")
            raise Exception("No hands detected in the image.")
            pass