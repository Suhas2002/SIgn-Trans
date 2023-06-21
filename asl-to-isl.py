from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import streamlit as st
import torch
from PIL import Image
from bokeh.models.widgets import Button
from bokeh.models import CustomJS

st.title("Two Way Sign Translator")


cap = cv2.VideoCapture(0)  # For Webcam
cap.set(3, 640)
cap.set(4, 480)
# cap = cv2.VideoCapture("../Videos/motorbikes.mp4")  # For Video
frame_placeholder = st.empty()
def text_to_speech(text):
    tts_button = Button(label="Speak", width=100)
    tts_button.js_on_event("button_click", CustomJS(code=f"""
    var u = new SpeechSynthesisUtterance();
    u.text = "{text}";
    u.lang = 'en-US';

    speechSynthesis.speak(u);
    """))
    st.bokeh_chart(tts_button)


# Add a "Stop" button and store its state in a variable
stop_button_pressed = st.button("Stop")

show_isl = st.button("Show ISL Sign")

device = torch.device('cpu')

model = YOLO("F:/Suhas Sem 6 Github/Sign-Trans/yolov8/asl to text/aslbest.pt")
# !yolo task=detect mode=predict model=/content/American-Sign-Language-Letters-1/runs/detect/train/weights/best.pt conf=0.25 source={dataset.location}/test/images save=True
classNames = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K",
              "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
              ]


# prev_frame_time = 0
# new_frame_time = 0

while True:
    new_frame_time = time.time()
    success, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = model(frame, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:

            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(frame, (x1, y1, w, h))
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])

            cvzone.putTextRect(frame, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

            frame_placeholder.image(frame, channels="RGB")

            if show_isl:
                if classNames[cls] == "A":
                    result = Image.open(r"F:\Suhas Sem 6 Github\Sign-Trans\yolov8\ISL_images\A.jpg")
                    st.image(result, caption='A',  width=200)
                    text_to_speech("A")
                    time.sleep(3)

                elif classNames[cls] == "B":
                    result = Image.open(r"F:\Suhas Sem 6 Github\Sign-Trans\yolov8\ISL_images\B.jpg")
                    st.image(result, caption='B',  width=200)
                    text_to_speech("B")
                    time.sleep(3)

                elif classNames[cls] == "C":
                    result = Image.open(r"F:\Suhas Sem 6 Github\Sign-Trans\yolov8\ISL_images\C.jpg")
                    st.image(result, caption='C',  width=200)
                    text_to_speech("C")
                    time.sleep(3)

                elif classNames[cls] == "D":
                    result = Image.open(r"F:\Suhas Sem 6 Github\Sign-Trans\yolov8\ISL_images\D.jpg")
                    st.image(result, caption='D',  width=200)
                    text_to_speech("D")
                    time.sleep(3)

                elif classNames[cls] == "E":
                    result = Image.open(r"F:\Suhas Sem 6 Github\Sign-Trans\yolov8\ISL_images\E.jpg")
                    st.image(result, caption='E',  width=200)
                    text_to_speech("E")
                    time.sleep(3)

                elif classNames[cls] == "F":
                    result = Image.open(r"F:\Suhas Sem 6 Github\Sign-Trans\yolov8\ISL_images\F.jpg")
                    st.image(result, caption='F',  width=200)
                    text_to_speech("F")
                    time.sleep(3)

                elif classNames[cls] == "G":
                    result = Image.open(r"F:\Suhas Sem 6 Github\Sign-Trans\yolov8\ISL_images\G.jpg")
                    st.image(result, caption='G', width=200)
                    text_to_speech("G")
                    time.sleep(3)

                elif classNames[cls] == "H":
                    result = Image.open(r"F:\Suhas Sem 6 Github\Sign-Trans\yolov8\ISL_images\H.jpg")
                    st.image(result, caption='H', width=200)
                    text_to_speech("H")
                    time.sleep(3)

                elif classNames[cls] == "I":
                    result = Image.open(r"F:\Suhas Sem 6 Github\Sign-Trans\yolov8\ISL_images\I.jpg")
                    st.image(result, caption='I', width=200)
                    text_to_speech("I")
                    time.sleep(3)

                elif classNames[cls] == "J":
                    result = Image.open(r"F:\Suhas Sem 6 Github\Sign-Trans\yolov8\ISL_images\J.jpg")
                    st.image(result, caption='J', width=200)
                    text_to_speech("J")
                    time.sleep(3)

                elif classNames[cls] == "K":
                    result = Image.open(r"F:\Suhas Sem 6 Github\Sign-Trans\yolov8\ISL_images\K.jpg")
                    st.image(result, caption='K', width=200)
                    text_to_speech("K")
                    time.sleep(3)

                elif classNames[cls] == "L":
                    result = Image.open(r"F:\Suhas Sem 6 Github\Sign-Trans\yolov8\ISL_images\L.jpg")
                    st.image(result, caption='L', width=200)
                    text_to_speech("L")
                    time.sleep(3)

                elif classNames[cls] == "M":
                    result = Image.open(r"F:\Suhas Sem 6 Github\Sign-Trans\yolov8\ISL_images\M.jpg")
                    st.image(result, caption='M', width=200)
                    text_to_speech("M")
                    time.sleep(3)

                elif classNames[cls] == "N":
                    result = Image.open(r"F:\Suhas Sem 6 Github\Sign-Trans\yolov8\ISL_images\N.jpg")
                    st.image(result, caption='N', width=200)
                    text_to_speech("N")
                    time.sleep(3)

                elif classNames[cls] == "O":
                    result = Image.open(r"F:\Suhas Sem 6 Github\Sign-Trans\yolov8\ISL_images\O.jpg")
                    st.image(result, caption='O', width=200)
                    text_to_speech("O")
                    time.sleep(3)

                elif classNames[cls] == "P":
                    result = Image.open(r"F:\Suhas Sem 6 Github\Sign-Trans\yolov8\ISL_images\P.jpg")
                    st.image(result, caption='P', width=200)
                    text_to_speech("P")
                    time.sleep(3)

                elif classNames[cls] == "Q":
                    result = Image.open(r"F:\Suhas Sem 6 Github\Sign-Trans\yolov8\ISL_images\Q.jpg")
                    st.image(result, caption='Q', width=200)
                    text_to_speech("Q")
                    time.sleep(3)

                elif classNames[cls] == "R":
                    result = Image.open(r"F:\Suhas Sem 6 Github\Sign-Trans\yolov8\ISL_images\R.jpg")
                    st.image(result, caption='R', width=200)
                    text_to_speech("R")
                    time.sleep(3)

                elif classNames[cls] == "S":
                    result = Image.open(r"F:\Suhas Sem 6 Github\Sign-Trans\yolov8\ISL_images\S.jpg")
                    st.image(result, caption='S', width=200)
                    text_to_speech("S")
                    time.sleep(3)

                elif classNames[cls] == "T":
                    result = Image.open(r"F:\Suhas Sem 6 Github\Sign-Trans\yolov8\ISL_images\T.jpg")
                    st.image(result, caption='T', width=200)
                    text_to_speech("T")
                    time.sleep(3)

                elif classNames[cls] == "U":
                    result = Image.open(r"F:\Suhas Sem 6 Github\Sign-Trans\yolov8\ISL_images\U.jpg")
                    st.image(result, caption='U', width=200)
                    text_to_speech("U")
                    time.sleep(3)

                elif classNames[cls] == "V":
                    result = Image.open(r"F:\Suhas Sem 6 Github\Sign-Trans\yolov8\ISL_images\V.jpg")
                    st.image(result, caption='V', width=200)
                    text_to_speech("V")
                    time.sleep(3)

                elif classNames[cls] == "W":
                    result = Image.open(r"F:\Suhas Sem 6 Github\Sign-Trans\yolov8\ISL_images\W.jpg")
                    st.image(result, caption='W',  width=200)
                    text_to_speech("W")
                    time.sleep(3)

                elif classNames[cls] == "X":
                    result = Image.open(r"F:\Suhas Sem 6 Github\Sign-Trans\yolov8\ISL_images\X.jpg")
                    st.image(result, caption='X', width=200)
                    text_to_speech("X")
                    time.sleep(3)

                elif classNames[cls] == "Y":
                    result = Image.open(r"F:\Suhas Sem 6 Github\Sign-Trans\yolov8\ISL_images\Y.jpg")
                    st.image(result, caption='Y', width=200)
                    text_to_speech("Y")
                    time.sleep(3)

                elif classNames[cls] == "Z":
                    result = Image.open(r"F:\Suhas Sem 6 Github\Sign-Trans\yolov8\ISL_images\Z.jpg")
                    st.image(result, caption='Z', width=200)
                    text_to_speech("Z")
                    time.sleep(3)

                else:
                    e = RuntimeError('Sign not found \n Try Again!')
                    st.exception(e)
                    time.sleep(3)
            if cv2.waitKey(1) & 0xFF == ord("q") or stop_button_pressed:
                break
    # fps = 1 / (new_frame_time - prev_frame_time)
    # prev_frame_time = new_frame_time
    # print(fps)

    cv2.imshow("Image", frame)
    cv2.waitKey(1)
