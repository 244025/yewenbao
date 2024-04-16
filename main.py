import cv2
import numpy as np
import os
import shutil
import threading
import tkinter as tk
from PIL import Image, ImageTk

# 首先读取config文件，第一行代表当前已经储存的人名个数，接下来每一行是（id，name）标签和对应的人名
id_dict = {}  # 字典里存的是id——name键值对
Total_face_num = 999  # 已经被识别有用户名的人脸个数

def init():  
    # 将config文件内的信息读入到字典中
    f = open('config.txt')
    global Total_face_num
    Total_face_num = int(f.readline())

    for i in range(int(Total_face_num)):
        line = f.readline().strip()
        if line:
            id_name = line.split(' ')
            id_dict[int(id_name[0])] = id_name[1]

init()

# 加载OpenCV人脸检测分类器Haar
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# 准备好识别方法LBPH方法
recognizer = cv2.face.LBPHFaceRecognizer_create()

# 打开标号为0的摄像头
camera = cv2.VideoCapture(0)  
success, img = camera.read()  
W_size = 0.1 * camera.get(3)
H_size = 0.1 * camera.get(4)

system_state_lock = 0  

def Get_new_face():
    print("正在从摄像头录入新人脸信息 \n")

    # 存在目录data就清空，不存在就创建，确保最后存在空的data目录
    filepath = "data"
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    else:
        shutil.rmtree(filepath)
        os.mkdir(filepath)

    sample_num = 0  

    while True: 
        global success
        global img  
        success, img = camera.read()

        if success is True:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            break

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + w), (255, 0, 0))
            sample_num += 1
            cv2.imwrite("./data/User." + str(Total_face_num) + '.' + str(sample_num) + '.jpg', gray[y:y + h, x:x + w])

        pictur_num = 1000
        cv2.waitKey(1)
        if sample_num > pictur_num:
            break
        else:
            l = int(sample_num / pictur_num * 50)
            r = int((pictur_num - sample_num) / pictur_num * 50)
            print("\r" + "%{:.1f}".format(sample_num / pictur_num * 100) + "=" * l + "->" + "_" * r, end="")
            var.set("%{:.1f}".format(sample_num / pictur_num * 100))  
            window.update()  

def Train_new_face():
    print("\n正在训练")
    path = 'data'

    recog = recognizer

    faces, ids = get_images_and_labels(path)

    recog.train(faces, np.array(ids))

    yml = str(Total_face_num) + ".yml"
    rec_f = open(yml, "w+")
    rec_f.close()
    recog.save(yml)

def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    face_samples = []
    ids = []

    for image_path in image_paths:
        img = Image.open(image_path).convert('L')
        img_np = np.array(img, 'uint8')

        if os.path.split(image_path)[-1].split(".")[-1] != 'jpg':
            continue

        id = int(os.path.split(image_path)[-1].split(".")[1])

        detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        faces = detector.detectMultiScale(img_np)

        for (x, y, w, h) in faces:
            face_samples.append(img_np[y:y + h, x:x + w])
            ids.append(id)
    return face_samples, ids

def write_config():
    print("训练结束")
    f = open('config.txt', "a")
    T = Total_face_num
    f.write(str(T) + " User" + str(T) + " \n")
    f.close()
    id_dict[T] = "User" + str(T)

    f = open('config.txt', 'r+')
    flist = f.readlines()
    flist[0] = str(int(flist[0]) + 1) + " \n"
    f.close()

    f = open('config.txt', 'w+')
    f.writelines(flist)
    f.close()

def scan_face():
    for i in range(Total_face_num):
        i += 1
        yml = str(i) + ".yml"
        print("\n本次:" + yml)  
        recognizer.read(yml)

        ave_poss = 0
        for times in range(10):  
            times += 1
            cur_poss = 0
            global success
            global img

            global system_state_lock
            while system_state_lock == 2:
                print("\r刷脸被录入面容阻塞", end="")
                pass

            success, img = camera.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(int(W_size), int(H_size))
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                idnum, confidence = recognizer.predict(gray[y:y + h, x:x + w])
                conf = confidence
                if confidence < 100:  
                    if idnum in id_dict:
                        user_name = id_dict[idnum]
                    else:
                        user_name = "Untagged user:" + str(idnum)
                    confidence = round(100 - confidence)
                else:  
                    user_name = "unknown"
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, str(user_name), (x + 5, y - 5), font, 1, (0, 0, 255), 1)
                cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (0, 0, 0), 1)
                print("conf=" + str(conf), end="\t")
                if 15 > conf > 0:
                    cur_poss = 1  
                elif 60 > conf > 35:
                    cur_poss = 1  
                else:
                    cur_poss = 0  

            k = cv2.waitKey(1)
            if k == 27:
                cv2.destroyAllWindows()
                break

            ave_poss += cur_poss

        if ave_poss >= 5:  
            return i

    return  0


def f_scan_face_thread():
    var.set('刷脸')
    ans = scan_face()
    if ans == 0:
        print("最终结果：识别失败")
        var.set("最终结果：识别失败")
    else:
        ans_name = "最终结果：" + str(ans) + id_dict[ans]
        print(ans_name)
        var.set(ans_name)

    global system_state_lock
    print("锁被释放0")
    system_state_lock = 0  

def f_scan_face():
    global system_state_lock
    print("\n当前锁的值为：" + str(system_state_lock))
    if system_state_lock == 1:
        print("阻塞，因为正在刷脸")
        return 0
    elif system_state_lock == 2:
        print("\n刷脸被录入面容阻塞\n")
        return 0
    system_state_lock = 1
    p = threading.Thread(target=f_scan_face_thread)
    p.setDaemon(True)  
    p.start()

def f_rec_face_thread():
    var.set('录入')
    cv2.destroyAllWindows()
    global Total_face_num
    Total_face_num += 1
    Get_new_face()  
    print("采集完毕，开始训练")
    global system_state_lock  
    print("锁被释放0")
    system_state_lock = 0

    Train_new_face()  
    write_config()  

def f_rec_face():
    global system_state_lock
    print("当前锁的值为：" + str(system_state_lock))
    if system_state_lock == 2:
        print("阻塞，因为正在录入面容")
        return 0
    else:
        system_state_lock = 2  
        print("改为2", end="")
        print("当前锁的值为：" + str(system_state_lock))

    p = threading.Thread(target=f_rec_face_thread)
    p.setDaemon(True)  
    p.start()

def f_exit():  
    exit()

window = tk.Tk()
window.title('yy\' Face_rec 3.0')   
window.geometry('1080x720')  

var = tk.StringVar()
l = tk.Label(window, textvariable=var, bg='green', fg='white', font=('Arial', 12), width=50, height=4)
l.pack()  

# 添加一个文本框用于实时显示日志信息
log_text = tk.Text(window, width=80, height=10)
log_text.place(x=250, y=500 )

button_a = tk.Button(window, text='开始刷脸', font=('Arial', 12), width=10, height=2, command=f_scan_face)
button_a.place(x=800, y=120)

button_b = tk.Button(window, text='录入人脸', font=('Arial', 12), width=10, height=2, command=f_rec_face)
button_b.place(x=800, y=220)

button_c = tk.Button(window, text='退出', font=('Arial', 12), width=10, height=2, command=f_exit)
button_c.place(x=800, y=320)

panel = tk.Label(window, width=500, height=350)  
panel.place(x=10, y=100)  

window.config(cursor="arrow")

def video_loop():  
    global success
    global img
    if success:
        cv2.waitKey(1)
        cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)  
        current_image = Image.fromarray(cv2image)  
        imgtk = ImageTk.PhotoImage(image=current_image)
        panel.imgtk = imgtk
        panel.config(image=imgtk)
        window.after(1, video_loop)

video_loop()

# 将日志信息输出到文本框中
def print_to_log(message):
    log_text.insert(tk.END, message + '\n')
    log_text.see(tk.END)  # 滚动到最后一行


window.mainloop()



















'''
def real_time_show():
    global success
    global img
    if success:
        show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        showImage = Image.fromarray(show)
        showImage = ImageTk.PhotoImage(showImage)
        label.config(image=showImage)
        label.image = showImage
        window.update()


# 添加输出到文本框
def add_output(text):
    output_text.insert(tk.END, text + "\n")
    output_text.see(tk.END)  # 滚动文本框以显示最新的输出


# 启动两个线程
t_scan_face = threading.Thread(target=f_scan_face_thread)
t_rec_face = threading.Thread(target=f_rec_face_thread)

t_scan_face.setDaemon(True)
t_rec_face.setDaemon(True)

t_scan_face.start()
t_rec_face.start()

# 实时显示
while True:
    real_time_show()
'''