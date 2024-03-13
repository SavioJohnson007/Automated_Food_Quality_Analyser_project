import os.path
import tkinter as tk
import time
import ttkbootstrap as ttk
import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
import serial
import cv2
from statistics import mode
import threading
import pyttsx3
from sklearn.tree import DecisionTreeClassifier
import serial.tools.list_ports

global thread
global c, s, r,ser_arm
flag = False

voice = pyttsx3.init()


def comp_check():
    global stop_event

    try:
        # cap = cv2.VideoCapture(1)
        # cam_connected = cap.isOpened()
        # cap.release()
        # if cam_connected:
        #     c_stat.set("connected")
        #     cam_stat.config(foreground='light green')
        # else:
        #     c_stat.set("disconnected")
        #     cam_stat.config(foreground='red')

        ports = serial.tools.list_ports.comports()
        available_ports = [p.device for p in ports]

        if 'COM7' in available_ports:
            s_stat.set("online")
            spec_stat.config(foreground='light green')
        else:
            s_stat.set("offline")
            spec_stat.config(foreground='red')
        if 'COM8' in available_ports:
            r_stat.set("online")
            arm_stat.config(foreground='light green')
        else:
            r_stat.set("offline")
            arm_stat.config(foreground='red')
        # if (cam_connected) & ('COM7' in available_ports):
        if 'COM7' in available_ports:
            if not flag:
                start_button.config(state=tk.NORMAL)
        else:
            stop_process()
            start_button.config(state=tk.DISABLED)

    except:
        pass

    window.after(1000, comp_check)


def send_to_arm(f, t, m, s):
    global ser_arm
    try:

        m_arm = f + "," + str(t) + "," + m + "," + s
        print(m_arm)
        ser_arm.write(m_arm.encode())
        time.sleep(2)
    except:
        pass


def process(event):
    global flag,ser_arm

    flag = True

    try:
        ser = serial.Serial('COM8', 115200)
        time.sleep(2)
        try :
           ser_arm = serial.Serial('COM7', 115200)
           time.sleep(2)
        except :
            pass

        while not event.is_set():
            n = 1

            count = 0
            ob_arr = []
            thres = 0.5
            NMSThresh = 0.1

            classNames = []
            classFile = r'Data\coco.names'
            with open(classFile, 'rt') as f:
                classNames = f.read().rstrip('\n').split('\n')

            configPath = r'Data\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
            weightsPath = r'Data\frozen_inference_graph.pb'

            net = cv2.dnn_DetectionModel(weightsPath, configPath)
            net.setInputSize(320, 320)
            net.setInputScale(1.0 / 127.5)
            net.setInputMean((127.5, 127.5, 127.5))
            net.setInputSwapRB(True)

            def spectral_analysis(Fruit):

                def read_serial():

                    st_array = np.zeros(6)
                    r = ser.readline()
                    r = r.decode().strip().rstrip()
                    while r != 'ready':
                        r = ser.readline()
                        r = r.decode().strip().rstrip()
                    if r == 'ready':
                        stat_text.set("Acquiring spectral data...")
                        for i in range(10):
                            ser.write("analysing".encode())
                            b = ser.readline()
                            string_n = b.decode()
                            string = string_n.rstrip()
                            my_array = string.split(",")
                            val_array = [float(x) for x in my_array]
                            st_array = np.vstack((st_array, val_array))
                        time.sleep(1)
                        ser.write("over".encode())
                        stn_array = np.delete(st_array, 0, axis=0)
                        return stn_array

                filename = 'Data\\' + Fruit + '.csv'

                if os.path.exists(filename):
                    ser.write("Take".encode())
                    data = pd.read_csv(filename)
                else:
                    ser.write("Skip".encode())
                    stat_text.set("No dataset available for " + Fruit)
                    text_label.config(foreground='red')
                    text_label.update()
                    voice.say("No dataset available for " + Fruit)
                    voice.runAndWait()
                    time.sleep(3)
                    text_label.config(foreground='white')
                    text_label.update()

                    return "Not Available", "Cannot be predicted", "Cannot be predicted"

                X = data.iloc[:, :-3].values
                y_tss = data.iloc[:, -3:-2].values
                y_maturity = data.iloc[:, -2:-1].values
                y_sweetness = data.iloc[:, -1:].values

                X_new = np.mean(read_serial(), axis=0).reshape(1, -1)

                def predict_tss():

                    n_components = 6
                    plsr = PLSRegression(n_components=n_components)

                    plsr.fit(X, y_tss)
                    y_pred = plsr.predict(X_new)

                    y_pred_tss = y_pred[0, 0]
                    y_pred_tss = round(y_pred_tss, 2)

                    return y_pred_tss

                def predict_sweetness_maturity():
                    clf_maturity = DecisionTreeClassifier(random_state=42)
                    clf_sweetness = DecisionTreeClassifier(random_state=42)

                    clf_maturity.fit(X, y_maturity)
                    clf_sweetness.fit(X, y_sweetness)

                    pred_ripeness = clf_maturity.predict(X_new)
                    pred_sweetness = clf_sweetness.predict(X_new)

                    return pred_ripeness[0], pred_sweetness[0]

                TSS = predict_tss()
                MATURITY, SWEETNESS = predict_sweetness_maturity()

                return TSS, MATURITY, SWEETNESS

            def getObjects(img, draw=True, objects=[]):
                classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=NMSThresh)

                if len(objects) == 0: objects = classNames
                objectInfo = []

                if len(classIds) != 0:

                    for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                        className = classNames[classId - 1]
                        if className in objects:
                            objectInfo = className
                            if draw:
                                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                                cv2.putText(img, className.upper(), (box[0] + 10, box[1] + 30),
                                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                                cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                return img, objectInfo

            cap = cv2.VideoCapture(1)
            cap.set(3, 640)
            cap.set(4, 480)
            cap.set(10, 70)
            ser.write("Start".encode())
            stat_text.set("Place the Fruit")
            while not event.is_set():
                success, img = cap.read()
                result, objectInfo = getObjects(img, objects=['orange', 'apple'])
                # cv2.imshow("Cam Live Feed", img)
                # cv2.waitKey(1)

                if (objectInfo != []) & (count < 50):
                    stat_text.set("Determining fruit ...")

                    count = count + 1
                    ob_arr.append(objectInfo)
                elif objectInfo != []:
                    fruit = mode(ob_arr)
                    stat_text.set("Fruit determined")
                    start_button.config(state=tk.DISABLED)
                    stop_button.config(state=tk.DISABLED)
                    fruit_op_string.set(fruit)
                    progress.step(50)
                    progress.update()
                    voice.say("The fruit is : " + fruit)
                    voice.runAndWait()
                    count = 0
                    ob_arr = []
                    tss, maturity, sweetness = spectral_analysis(fruit)
                    if tss != "Not Available":
                        stat_text.set("Spectral analysis complete")
                    tss_op_string.set(tss)
                    sweetness_op_string.set(sweetness)
                    maturity_op_string.set(maturity)
                    voice.say("The TSS is " + str(tss))
                    voice.runAndWait()
                    voice.say("The sweetness is " + str(sweetness))
                    voice.runAndWait()
                    voice.say("The maturity of the fruit is  " + str(maturity))
                    voice.runAndWait()
                    send_to_arm(fruit, tss, maturity, sweetness)
                    progress.step(49)
                    progress.update()
                    time.sleep(4)
                    ser.write("Start".encode())
                    progress.step(-progress['value'])
                    fruit_op_string.set("")
                    tss_op_string.set("")
                    maturity_op_string.set("")
                    sweetness_op_string.set("")
                    num_box.insert(tk.END, str(n) + "\n\n")
                    log_box.insert(tk.END, fruit + "  ;  TSS = " + str(
                        tss) + "  ;  Sweetness = " + sweetness + "  ;  Maturity = " + maturity + "\n\n")
                    n = n + 1
                    start_button.config(state=tk.DISABLED)
                    stop_button.config(state=tk.NORMAL)

                else:
                    pass
            ser.write("Stop".encode())
            cap.release()
            cv2.destroyAllWindows()
    except:
        flag = False
        pass

    flag = False


def start_process():
    global thread, stop_event

    stop_event = threading.Event()
    thread = threading.Thread(target=process, args=(stop_event,))
    thread.start()
    stat_text.set("Starting...")
    text_label.config(foreground='white')
    text_label.update()
    start_button.config(state=tk.DISABLED)
    stop_button.config(state=tk.NORMAL)


def stop_process():
    global stop_event,ser_arm
    ser_arm.close()
    if thread and thread.is_alive():
        stop_event.set()
        thread.join()
        time.sleep(1)
        stat_text.set("")
        fruit_op_string.set("")
        tss_op_string.set("")
        sweetness_op_string.set("")
        maturity_op_string.set("")
        text_label.update()
        start_button.config(state=tk.NORMAL)
        stop_button.config(state=tk.DISABLED)


window = ttk.Window(themename='superhero')
window.title('Spectroscopic Food Quality Analyser(for Fruits)')
window.geometry('950x950')
window.wm_state('zoomed')
window.attributes('-fullscreen', False)

title_frame = ttk.Frame(master=window)
title_label = ttk.Label(master=title_frame, text='FOOD QUALITY ANALYSER', font=('Gill Sans Ultra Bold', 30),
                        foreground='light blue')
title_label.pack(pady=10, anchor='center')
credit_label = ttk.Label(title_frame,text ='SpectraSense',font=('Times New Roman',20,'bold'),foreground='black')
credit_label.pack(pady=10)
title_frame.pack(anchor='center',pady=50)

comp_status = ttk.Frame(master=window)
# cam_frame = ttk.Frame(master=comp_status)
# c_stat = tk.StringVar()
# cam = ttk.Label(master=cam_frame, text="Camera:", font=('Cascadia Code', 15, 'bold'))
# cam.pack(side='left')
# cam_stat = ttk.Label(master=cam_frame, textvariable=c_stat, font=('Cascadia Code', 15, 'bold'))
# cam_stat.pack(side='right')
# cam_frame.pack(side='left')
spec_frame = ttk.Frame(master=comp_status)
s_stat = tk.StringVar()
spec = ttk.Label(master=spec_frame, text="    Spectroscopic Unit:", font=('Cascadia Code', 15, 'bold'))
spec.pack(side='left')
spec_stat = ttk.Label(master=spec_frame, textvariable=s_stat, font=('Cascadia Code', 15, 'bold'))
spec_stat.pack(side='right')
spec_frame.pack(side='left')
arm_frame = ttk.Frame(master=comp_status)
r_stat = tk.StringVar()
arm = ttk.Label(master=arm_frame, text="    Robotic Arm:", font=('Cascadia Code', 15, 'bold'))
arm.pack(side='left')
arm_stat = ttk.Label(master=arm_frame, textvariable=r_stat, font=('Cascadia Code', 15, 'bold'))
arm_stat.pack(side='right')
arm_frame.pack(side='right')
comp_status.pack(anchor='center')

input_frame = ttk.Frame(master=window, width=300, height=100)
button_style = ttk.Style()
button_style.configure('W1.TButton', font=('Cascadia Code', 20, 'bold'), foreground='white', background='light green')
button_style.map('TButton', foreground=[('active', '!disabled', 'white')])
start_button = ttk.Button(master=input_frame, text='Start', style='W1.TButton', command=start_process, width=5,
                          state=tk.DISABLED)
start_button.pack(pady=20, side='left', padx=10)
button_style.configure('W2.TButton', font=('Cascadia Code', 20, 'bold'), foreground='white', background='red')
stop_button = ttk.Button(master=input_frame, text='Stop', style='W2.TButton', command=stop_process, width=5,
                         state=tk.DISABLED)
stop_button.pack(pady=20, side='left', padx=20)

input_frame.pack(pady=30, anchor='center')

output_frame = ttk.Frame(master=window)

fruit_frame = ttk.Frame(master=output_frame)
fruit_op_string = tk.StringVar()
fruit_label = ttk.Label(master=fruit_frame, text='Fruit : ', font=('OCR A Extended', 24, 'bold'))
fruit_label.pack(side='left')
fruit_op = ttk.Label(master=fruit_frame, textvariable=fruit_op_string, font=('Cascadia Mono', 26, 'bold'),
                     foreground='orange')
fruit_op.pack(side='right')
fruit_frame.pack(pady=10)

tss_frame = ttk.Frame(master=output_frame)
tss_op_string = tk.StringVar()
tss_label = ttk.Label(master=tss_frame, text='TSS : ', font=('OCR A Extended', 24, 'bold'))
tss_label.pack(side='left')
tss_op = ttk.Label(master=tss_frame, textvariable=tss_op_string, font=('Cascadia Mono', 26, 'bold'),
                   foreground='orange')
tss_op.pack(side='right')
tss_frame.pack(pady=10)

sweetness_frame = ttk.Frame(master=output_frame)
sweetness_op_string = tk.StringVar()
sweetness_label = ttk.Label(master=sweetness_frame, text='Sweetness : ', font=('OCR A Extended', 24, 'bold'))
sweetness_label.pack(side='left')
sweetness_op = ttk.Label(master=sweetness_frame, textvariable=sweetness_op_string, font=('Cascadia Mono', 26, 'bold'),
                         foreground='orange')
sweetness_op.pack(side='right')
sweetness_frame.pack(pady=10)

maturity_frame = ttk.Frame(master=output_frame)
maturity_op_string = tk.StringVar()
maturity_label = ttk.Label(master=maturity_frame, text='Maturity : ', font=('OCR A Extended', 24, 'bold'))
maturity_label.pack(side='left')
maturity_op = ttk.Label(master=maturity_frame, textvariable=maturity_op_string, font=('Cascadia Mono', 26, 'bold'),
                        foreground='orange')
maturity_op.pack(side='right')
maturity_frame.pack(pady=10)

output_frame.pack(anchor='center')

bar_style = ttk.Style()
bar_style.configure('W.Horizontal.TProgressbar', background='blue')
progress = ttk.Progressbar(window, maximum=100, length=320, mode='determinate', style='W.Horizontal.TProgressbar')
progress.pack(pady=10)

stat_text = ttk.StringVar()
text_label = ttk.Label(window, textvariable=stat_text, foreground='white')
text_label.pack(anchor='center')

log = ttk.Frame(master=window)
scrollbar = ttk.Scrollbar(master=log)
scrollbar.pack(side=ttk.RIGHT, fill=ttk.Y)
num_box = ttk.Text(master=log, height='12', width='2',
                   yscrollcommand=lambda *args: scroll_both(num_box, log_box, *args), font=("Helvetica", 10))
num_box.pack(side=ttk.LEFT, fill=ttk.Y)
log_box = ttk.Text(master=log, height='12', width='85',
                   yscrollcommand=lambda *args: scroll_both(log_box, num_box, *args), font=("Helvetica", 10))
log_box.pack(fill=tk.BOTH)

scrollbar.config(command=lambda *args: [num_box.yview(*args), log_box.yview(*args)])
log.pack(pady=10, anchor='center')


def scroll_both(textbox, other_textbox, *args):
    """Scroll two textboxes in sync"""
    # update text boxes before scrolling
    textbox.update()
    other_textbox.update()

    # get the scroll position of the current textbox
    pos = float(args[0])

    # set the scroll position of the other textbox to match
    other_textbox.yview_moveto(pos)
    scrollbar.set(*args)


comp_check()
window.mainloop()
