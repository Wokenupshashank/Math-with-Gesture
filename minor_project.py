import cvzone
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import google.generativeai as genai
from PIL import Image
import streamlit as st

# Configure Streamlit
st.set_page_config(layout="wide")
st.image('background.png')

# Define columns for the Streamlit layout
col1, col2 = st.columns([3, 2])  
with col1:
    run = st.checkbox('Run', value=True)
    FRAME_WINDOW = st.image([])

with col2:
    st.title("Solution")
    output_text_area = st.subheader("")

# Initialize the Generative AI model
genai.configure(api_key="AIzaSyCG3bW9sn4Q7k5zNojocLHKIAPozZ7fb5c")
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize the webcam to capture video
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Initialize the HandDetector class
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.8, minTrackCon=0.5)

def getHandInfo(img):
    """Find hands in the current frame and return information."""
    hands, img = detector.findHands(img, draw=False, flipType=True)
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        print(fingers)  # For debugging purposes
        return fingers, lmList
    return None

def draw(info, prev_pos, canvas):
    """Draw on the canvas based on hand gestures."""
    fingers, lmList = info
    current_pos = None
    if fingers == [0, 1, 0, 0, 0]:
        current_pos = lmList[8][0:2]
        if prev_pos is None:
            prev_pos = current_pos
        cv2.line(canvas, current_pos, prev_pos, (120, 0, 205), 10)
    elif fingers == [1, 0, 0, 0, 0]:
        canvas = np.zeros_like(canvas)  # Correctly reset the canvas
    
    return current_pos, canvas

def sendToAI(model, canvas, fingers):
    """Send the canvas to the AI model and get the response."""
    if fingers == [1, 1, 1, 1, 0]:
        pil_image = Image.fromarray(canvas)
        response = model.generate_content(["Solve this math problem", pil_image])
        return response.text
    return ""

prev_pos = None
canvas = None

# Main loop for capturing and processing video frames
while True:
    success, img = cap.read()
    if not success or img is None:
        st.error("Error: Failed to capture image from camera.")
        continue
    
    img = cv2.flip(img, 1)
    
    if canvas is None:
        canvas = np.zeros_like(img)
    
    info = getHandInfo(img)
    if info:
        fingers, lmList = info
        prev_pos, canvas = draw(info, prev_pos, canvas)
        output_text = sendToAI(model, canvas, fingers)
    else:
        output_text = ""
    
    image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
    FRAME_WINDOW.image(image_combined, channels="BGR")
    
    if output_text:
        output_text_area.text(output_text)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Optionally exit on 'q' key
        break

cap.release()
cv2.destroyAllWindows()


