from tkinter import *
import cv2
from PIL import Image, ImageTk
import numpy as np
import pyttsx3
from translate import Translator

root = Tk()
root.geometry("900x900")
root.title("Sign Language To Text/Speech")

title = Label(root)
title.place(x=60, y=5)
title.configure(text="Sign Language To Text/Speech", font=("Courier", 30))

cam = Label(root)
cam.place(x=120, y=60, width=540, height=480)
cap = cv2.VideoCapture(0)

net = cv2.dnn.readNetFromDarknet("yolov3_custom.cfg", "yolov3_custom_last.weights")
classes = ['Hello', 'I Love You', 'No', 'Thank You', 'Yes']
label = None
sentence_array = []


def show_frame():
    _, frame = cap.read()
    frame = cv2.resize(frame, (540, 480))
    frame = cv2.flip(frame, 1)

    frame, a = process_frame(frame)

    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    cam.imgtk = imgtk
    cam.configure(image=imgtk)

    display_text.configure(text=a, font=("Courier", 30))

    cam.after(10, show_frame)


def process_frame(img):
    global label
    height, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

    net.setInput(blob)

    output_layers_name = net.getUnconnectedOutLayersNames()

    layerOutputs = net.forward(output_layers_name)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]
            if confidence > 0.7:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, .5, .4)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, .8, .4)
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label + " " + confidence, (x, y), font, 2, color, 2)
    return img, label


def text_to_speech():
    engine = pyttsx3.init()
    engine.say(label)
    engine.runAndWait()

def translate():
    translator = Translator(to_lang="Hindi")
    translation = translator.translate(label)
    display_translation.configure(text=translation, font=("Courier", 30))


field = Label(root, text="Text:")
field.place(x=60, y=570)
field.configure(font=("Courier", 30))

display_text = Label(root)
display_text.place(x=220, y=570)

display_translation = Label(root)
display_translation.place(x=520, y=570)

button1 = Button(root, command=text_to_speech)
button1.place(x=60, y=640, width=300, height=100)
button1.configure(text="Audio", font=("Courier", 30))

button1 = Button(root, command=translate)
button1.place(x=400, y=640, width=300, height=100)
button1.configure(text="Translate", font=("Courier", 30))

button2 = Button(root, command=root.destroy)
button2.place(x=760, y=640, width=300, height=100)
button2.configure(text="Quit", font=("Courier", 30))

show_frame()
root.mainloop()

