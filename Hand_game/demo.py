import cv2
import mediapipe as mp
import math
import keyboard
import time

# Load the hand landmark model
def calculate_angle(hand_landmarks):
    # Get landmarks for the index, middle, and ring fingers
    index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_finger = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_finger = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]

    # Calculate the angle between the fingers using the dot product and arccosine
    v1 = [index_finger.x - middle_finger.x, index_finger.y - middle_finger.y]
    v2 = [ring_finger.x - middle_finger.x, ring_finger.y - middle_finger.y]
    angle = math.degrees(math.acos(
        (v1[0] * v2[0] + v1[1] * v2[1]) / (math.sqrt(v1[0] ** 2 + v1[1] ** 2) * math.sqrt(v2[0] ** 2 + v2[1] ** 2))))

    return angle
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.9,
    min_tracking_confidence=0.9)

# Create a drawing object to draw landmarks and connections
mp_drawing = mp.solutions.drawing_utils

# Capture video from the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to RGB format
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hand landmarks
    results = hands.process(frame)

    # Convert the frame back to BGR format
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Draw landmarks on the frame
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Calculate the angle between finger landmarks
            angle = calculate_angle(hand_landmarks)
            #print(angle)
            # Show the angle on the frame
            cv2.putText(frame, f'Angle: {angle}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            

            # Press spacebar if angle is less than 30 degrees
            if angle < 70:
                keyboard.press('space')
                time.sleep(0.01)
                keyboard.release('space')
                cv2.putText(frame, 'Press Spacebar', (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if cv2.waitKey(1) & 0xFF == ord(' '):
                    print('Spacebar pressed')

    # Show the frame
    cv2.imshow('frame', frame)

    # Exit if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()


