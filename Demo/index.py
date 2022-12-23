import re
from database import datastore
from tkinter import *
import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
import pandas as pd
from tkinter import filedialog
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import module.zoom as zoom
from module.fftAlgorithm import fftenhencement
from module.histogram import histogram
import sys
from module.FingerprintImageEnhancer import FingerprintImageEnhancer
import cv2
from module.thinning import thinning
from module.minutiae import extract_minutiae_features

from pathlib import Path
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage

# đường dẫn cửa sổ 1
OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r".\assets\frame0")
ImageFile.LOAD_TRUNCATED_IMAGES = True


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


# đường dẫn cửa sổ add database
OUTPUT_PATH1 = Path("asset\frame1").parent
ASSETS_PATH1 = OUTPUT_PATH1 / Path(r".\assets\frame1")


def relative_to_assets1(path: str) -> Path:
    return ASSETS_PATH1 / Path(path)


def main():
    img_filename = path
    save_filename = 'histogram.png'

    histogram(img_filename, save_filename)

    # fft enhencement
    
    # dark_image = plt.imread('histogram.png')
    # dark_image = dark_image.astype(float)/255
    # complete_image = 'fft.png'

    #fftenhencement(img_filename, complete_image)
    
    # normalise the image and find a ROI
    image_enhancer = FingerprintImageEnhancer()         # Create object called image_enhancer
    if(len(sys.argv)<2):                                # load input image
        print('loading sample image')
        img_name = save_filename
        img = cv2.imread(img_name)
    elif(len(sys.argv) >= 2):
        img_name = sys.argv[1]
        img = cv2.imread(img_name)

    if(len(img.shape)>2):                               # convert image into gray if necessary
         img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    out = image_enhancer.enhance(img)     # run image enhancer
    image_enhancer.save_enhanced_image("enhancered.png")   # save output
    
    # Call thinning funtion
    thinning()
    
    # Call funtion minutiae
    img = cv2.imread('thinninged.png', 0)				# read the input image --> You can enhance the fingerprint image using the "fingerprint_enhancer" library
    FeaturesTerminations, FeaturesBifurcations = extract_minutiae_features(img, spuriousMinutiaeThresh=10, invertImage = False, showResult=True, saveResult = True)

# hàm chạy cùng lúc 2 hàm
def sequence(*functions):
    def func(*args, **kwargs):
        return_value = None
        for function in functions:
            return_value = function(*args, **kwargs)
        return return_value
    return func

# GUI
# chọn file window 1


def browsefiles():
    global path
    path = filedialog.askopenfilename(
        initialdir="/", title="Select A File", filetype=(("png files", ".png"), ("all files", "."), ("jpeg files", ".jpeg"),  ("jpg files", ".jpg"), ("tif files", ".tif")))
    print(path)

    if (path == ""):
        messagebox.showwarning(
            "showwarning", "Bạn chưa nhập ảnh nào, hãy nhập lại")
        path = relative_to_assets("user.png")
        return

    global photopath
    photopath = PhotoImage(file=path)
    showbutton()


def showbutton():
    buttonimg = Button(window,
                       cursor="hand2",
                       image=photopath,
                       borderwidth=0,
                       highlightthickness=0,
                       command=sequence(zooming, resize),
                       relief="flat"
                       )
    buttonimg.place(
        x=85,
        y=125,
        width=200,
        height=200
    )


def browsefiles2():
    global path
    path = filedialog.askopenfilename(
        initialdir="/", title="Select A File", filetype=(("png files", ".png"), ("jpeg files", ".jpeg"), ("jpg files", ".jpg"), ("tif files", ".tif")))
    entry_51.config(state="normal")
    entry_51.insert(0, path)
    entry_51.config(state="disabled")


# resize

def resize():

    image = Image.open(path)

    resized = image.resize((100, 100))
    resized.show()

# chọn ava window 2


def browsefiles3():
    global path1
    path1 = filedialog.askopenfilename(
        initialdir="/", title="Select A File", filetype=(("png files", ".png"), ("jpg files", ".jpg"), ("jpeg files", ".jpeg"), ("tif files", ".tif")))
    entry_11.config(state="normal")
    entry_11.insert(0, path1)
    entry_11.config(state="disabled")

# window 2


def zooming():
    app = zoom.MainWindow(Toplevel(), path=path)
    app.mainloop()


def openlabel():
    global window1
    window1 = Toplevel()
    window1.geometry("862x519")
    window1.configure(bg="#3A7FF6")
    window1.title("Adding To Database")
    window1.transient([window])

    canvas = Canvas(
        window1,
        bg="#3A7FF6",
        height=519,
        width=862,
        highlightthickness=0,
        relief="ridge"
    )

    canvas.place(x=0, y=0)
    canvas.create_rectangle(
        1.1368683772161603e-13,
        80.0,
        862.0000000000001,
        519.0,
        fill="#FCFCFC",
        outline="")

    # entry vân tay
    entry_image_11 = PhotoImage(
        file=relative_to_assets1("entry_1.png"))
    entry_bg_11 = canvas.create_image(
        666.4999999999999,
        360.0,
        image=entry_image_11
    )

    global entry_11
    entry_11 = Entry(
        window1,
        cursor="arrow",
        state="disabled",
        bd=0,
        bg="#DBDFEA",
        fg="#000716",
        highlightthickness=0
    )
    entry_11.place(
        x=593.9999999999999,
        y=345.0,
        width=145.0,
        height=28.0
    )

    canvas.create_text(
        49.999999999999886,
        116.0,
        anchor="nw",
        text="Tên :",
        fill="#3A7FF6",
        font=("Roboto Bold", 14 * -1)
    )

    canvas.create_text(
        49.999999999999886,
        116.0,
        anchor="nw",
        text="Tên :",
        fill="#3A7FF6",
        font=("Roboto Bold", 14 * -1)
    )

    canvas.create_text(
        49.999999999999886,
        116.0,
        anchor="nw",
        text="Tên :",
        fill="#3A7FF6",
        font=("Roboto Bold", 14 * -1)
    )

    canvas.create_text(
        49.999999999999886,
        313.0,
        anchor="nw",
        text="Học tại trường :",
        fill="#3A7FF6",
        font=("Roboto Bold", 14 * -1)
    )

    canvas.create_text(
        586.9999999999999,
        116.0,
        anchor="nw",
        text="Giới tính :",
        fill="#3A7FF6",
        font=("Roboto Bold", 14 * -1)
    )

    canvas.create_text(
        586.9999999999999,
        204.0,
        anchor="nw",
        text="Ngành :",
        fill="#3A7FF6",
        font=("Roboto Bold", 14 * -1)
    )

    canvas.create_text(
        586.9999999999999,
        313.0,
        anchor="nw",
        text="Ảnh vân tay :",
        fill="#3A7FF6",
        font=("Roboto Bold", 14 * -1)
    )

    canvas.create_text(
        345.9999999999999,
        313.0,
        anchor="nw",
        text="Ảnh hồ sơ :",
        fill="#3A7FF6",
        font=("Roboto Bold", 14 * -1)
    )

    # button xác nhận
    button_image_11 = PhotoImage(
        file=relative_to_assets1("button_1.png"))
    button_1 = Button(
        window1,
        cursor="hand2",
        image=button_image_11,
        borderwidth=0,
        highlightthickness=0,
        command=getinputvalue,
        relief="flat"
    )
    button_1.place(
        x=340.9999999999999,
        y=401.0,
        width=180.0,
        height=55.0
    )

    canvas.create_text(
        49.999999999999886,
        204.0,
        anchor="nw",
        text="Tuổi :",
        fill="#3A7FF6",
        font=("Roboto Bold", 14 * -1)
    )

    canvas.create_text(
        284.9999999999999,
        25.000000000000007,
        anchor="nw",
        text="Thêm dữ liệu vào database",
        fill="#FFFFFF",
        font=("Arial Bold", 24 * -1)
    )

    # entry học tại trường
    entry_image_21 = PhotoImage(
        file=relative_to_assets1("entry_2.png"))
    entry_bg_21 = canvas.create_image(
        134.4999999999999,
        360.0,
        image=entry_image_21
    )
    global entry_21
    entry_21 = Entry(
        window1,
        bd=0,
        bg="#3A7FF6",
        fg="#000716",
        highlightthickness=0
    )
    entry_21.place(
        x=61.999999999999886,
        y=345.0,
        width=145.0,
        height=28.0
    )

    # entry tuổi
    entry_image_31 = PhotoImage(
        file=relative_to_assets1("entry_3.png"))
    entry_bg_31 = canvas.create_image(
        666.4999999999999,
        260.0,
        image=entry_image_31
    )
    global entry_31
    entry_31 = Entry(
        window1,
        bd=0,
        bg="#3A7FF6",
        fg="#000716",
        highlightthickness=0
    )
    entry_31.place(
        x=593.9999999999999,
        y=245.0,
        width=145.0,
        height=28.0
    )
    box_value1 = StringVar()

    global combobox11
    combobox11 = ttk.Combobox(
        window1, width=17, textvariable=box_value1, state='readonly')
    for x in range(1, 120):
        combobox11['values'] = tuple(list(combobox11['values']) + [str(x)])

    combobox11.current(0)
    combobox11.place(
        x=53.9999999999999,
        y=245.0,
        width=165.0,
        height=28.0
    )

    box_value21 = StringVar()
    global combobox21
    combobox21 = ttk.Combobox(
        window1, width=17, textvariable=box_value21, state='readonly')
    combobox21['values'] = ('Nam', 'Nữ')
    combobox21.current(0)
    combobox21.place(
        x=583.9999999999999,
        y=145.0,
        width=165.0,
        height=28.0
    )

    # entry tên
    entry_image_41 = PhotoImage(
        file=relative_to_assets1("entry_4.png"))
    entry_bg_41 = canvas.create_image(
        134.4999999999999,
        168.0,
        image=entry_image_41
    )
    global entry_41
    entry_41 = Entry(
        window1,
        bd=0,
        bg="#3A7FF6",
        fg="#000716",
        highlightthickness=0
    )
    entry_41.place(
        x=61.999999999999886,
        y=153.0,
        width=145.0,
        height=28.0
    )

    # Ảnh hồ sơ
    entry_image_51 = PhotoImage(
        file=relative_to_assets1("entry_5.png"))
    entry_bg_51 = canvas.create_image(
        430.4999999999999,
        360.0,
        image=entry_image_51
    )
    global entry_51
    entry_51 = Entry(
        window1,
        cursor="arrow",
        state="disabled",
        bd=0,
        bg="#DBDFEA",
        fg="#000716",
        highlightthickness=0
    )
    entry_51.place(
        x=357.9999999999999,
        y=345.0,
        width=145.0,
        height=28.0
    )

    # button: ảnh hồ sơ
    button_image_21 = PhotoImage(
        file=relative_to_assets1("button_2.png"))
    button_21 = Button(
        window1,
        cursor="hand2",
        image=button_image_21,
        borderwidth=0,
        highlightthickness=0,
        command=browsefiles2,
        relief="flat"
    )
    button_21.place(
        x=487.9999999999999,
        y=354.0,
        width=14.0,
        height=12.0
    )

    # button: ảnh vân tay
    button_image_31 = PhotoImage(
        file=relative_to_assets1("button_3.png"))
    button_31 = Button(
        window1,
        image=button_image_31,
        borderwidth=0,
        highlightthickness=0,
        command=browsefiles3,
        relief="flat"
    )
    button_31.place(
        x=725.9999999999999,
        y=354.0,
        width=14.0,
        height=12.0
    )
    window1.resizable(False, False)
    window1.mainloop()


def getinputvalue():
    value_ten1 = entry_41.get()
    value_age = combobox11.get()
    value_gender = combobox21.get()
    value_imploy1 = entry_31.get()
    value_income1 = entry_21.get()
    value_ten = value_ten1.strip()
    value_path1 = entry_51.get()
    value_path2 = entry_11.get()
    value_imploy = value_imploy1.strip()
    value_income = value_income1.strip()

    regexname = "^[a-zA-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂẾưăạảấầẩẫậắằẳẵặẹẻẽềềểếỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễệỉịọỏốồổỗộớờởỡợụủứừỬỮỰỲỴÝỶỸửữựỳỵỷỹ\s\W|_]+$"
    regeximploy = "^[A-Za-z0-9 _]*[A-Za-z0-9][A-Za-z0-9 _]*$"
    regexincome = "^[A-Za-z0-9 _]*[A-Za-z0-9][A-Za-z0-9 _]*$"

    if (value_ten == "" or value_imploy == "" or value_income == "" or value_path1 == "" or value_path2 == ""):
        window1.destroy()
        tk.messagebox.showerror(
            title=None, message="Có trường để trống hãy xem lại")
    elif bool(re.search(regexname, value_ten)) == False:
        tk.messagebox.showerror(
            title=None, message="Trường tên nhập sai, hãy nhập lại")
    elif bool(re.search(regeximploy, value_imploy) == False):
        tk.messagebox.showerror(
            title=None, message="Trường ngành nhập sai, hãy nhập lại")
    elif bool(re.search(regexincome, value_income) == False):
        tk.messagebox.showerror(
            title=None, message="Trường nơi học nhập sai, hãy nhập lại")
    # elif bool(re.search(regex))
    else:
        datanow = datastore["info"]["identification"]

        lastvalue = datanow[-1]["index"]

        valueadd = {"index": lastvalue + 1,
                    "avatar": value_path1,
                    "fingerprint": value_path2,
                    "name": value_ten,
                    "age": value_age,
                    "study-at": value_income,
                    "gender": value_gender,
                    "speciality": value_imploy
                    }

        datanow.append(valueadd)
        tk.messagebox.showinfo("Cập nhật Database",
                               "Thành công cập nhật thông tin database")

    f = open("database.py", "r+")
    f.truncate(0)
    f.write("datastore = " + str(datastore))
    f.close()

# window


def addinfo():
    openlabel()


def normallabel():
    window.grab_release()


window = Tk()
window.title("ĐỒ ÁN XỬ LÍ ẢNH")
window.geometry("863x519")
window.configure(bg="#3A7FF6")

canvas = Canvas(
    window,
    bg="#3A7FF6",
    height=519,
    width=862,
    bd=0,
    highlightthickness=0,
    relief="ridge"
)
canvas.place(x=0, y=0)

canvas.create_text(
    637.9999999999999,
    101.0,
    anchor="nw",
    text="Tuổi :",
    fill="#FFFFFF",
    font=("Roboto Bold", 16 * -1)
)

canvas.create_text(
    394.9999999999999,
    265.0,
    anchor="nw",
    text="Ngành :",
    fill="#FFFFFF",
    font=("Roboto Bold", 16 * -1)
)

canvas.create_text(
    637.9999999999999,
    179.0,
    anchor="nw",
    text="Học tại :",
    fill="#FFFFFF",
    font=("Roboto Bold", 16 * -1)
)

canvas.create_text(
    393.9999999999999,
    187.0,
    anchor="nw",
    text="Giới tính :",
    fill="#FFFFFF",
    font=("Roboto Bold", 16 * -1)
)

canvas.create_text(
    392.9999999999999,
    101.0,
    anchor="nw",
    text="Tên :\n",
    fill="#FFFFFF",
    font=("Roboto Bold", 16 * -1)
)

canvas.create_text(
    140,
    70,
    anchor="nw",
    text="Avatar",
    fill="#ff9e17",
    font=("Franklin Gothic Demi", 30 * -1)
)


canvas.create_text(
    397.9999999999999,
    21.000000000000007,
    anchor="nw",
    text="Danh tính người được nhận diện\n",
    fill="#FFFFFF",
    font=("Roboto Bold", 24 * -1)
)

# button1: xác nhận
button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
button_1 = Button(
    cursor="hand2",
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=main,
    relief="flat"
)
button_1.place(
    x=379.9999999999999,
    y=408.0,
    width=180.0,
    height=55.0
)

# avatar
avata = PhotoImage(
    file=relative_to_assets("user.png"))
labelava = Label(image=avata, highlightbackground="#ff9e17",
                 highlightthickness=5)
labelava.place(
    x=80,
    y=120
)

# button 2: đầu vào
button_image_2 = PhotoImage(
    file=relative_to_assets("button_2.png"))
button_2 = Button(
    cursor="hand2",
    image=button_image_2,
    highlightthickness=0,
    command=browsefiles,
    relief="flat",
    borderwidth=1
)
button_2.place(
    x=90,
    y=407.0,
    width=180.0,
    height=55.0
)

# button 3: thêm database
button_image_3 = PhotoImage(
    file=relative_to_assets("button_3.png"))
button_3 = Button(
    cursor="hand2",
    image=button_image_3,
    borderwidth=0,
    highlightthickness=0,
    command=addinfo,
    relief="flat"
)
button_3.place(
    x=621.93,
    y=407.6,
    width=180.0,
    height=55.0
)

# entry: Tên
entry_image_1 = PhotoImage(
    file=relative_to_assets("entry_1.png"))
entry_bg_1 = canvas.create_image(
    468.4999999999999,
    150.0,
    image=entry_image_1
)
entry_1 = Entry(
    cursor="arrow",
    state="disabled",
    bd=0,
    bg="#7990E3",
    fg="#000716",
    highlightthickness=0
)
entry_1.place(
    x=404.9999999999999,
    y=137.0,
    width=127.0,
    height=24.0
)

# entry: học tại
entry_image_2 = PhotoImage(
    file=relative_to_assets("entry_2.png"))
entry_bg_2 = canvas.create_image(
    711.4999999999999,
    222.0,
    image=entry_image_2
)
entry_2 = Entry(
    cursor="arrow",
    state="disabled",
    bd=0,
    bg="#7990E3",
    fg="#000716",
    highlightthickness=0
)
entry_2.place(
    x=649.9999999999999,
    y=209.0,
    width=123.0,
    height=24.0
)

# entry ngành
entry_image_3 = PhotoImage(
    file=relative_to_assets("entry_3.png"))
entry_bg_3 = canvas.create_image(
    470.4999999999999,
    308.0,
    image=entry_image_3
)
entry_3 = Entry(
    cursor="arrow",
    state="disabled",
    bd=0,
    bg="#7990E3",
    fg="#000716",
    highlightthickness=0
)
entry_3.place(
    x=406.9999999999999,
    y=295.0,
    width=127.0,
    height=24.0
)

# entry: giới tính
entry_image_4 = PhotoImage(
    file=relative_to_assets("entry_4.png"))
entry_bg_4 = canvas.create_image(
    469.4999999999999,
    222.0,
    image=entry_image_4
)
entry_4 = Entry(
    cursor="arrow",
    state="disabled",
    bd=0,
    bg="#7990E3",
    fg="#000716",
    highlightthickness=0
)
entry_4.place(
    x=405.9999999999999,
    y=209.0,
    width=127.0,
    height=24.0
)

# entry: tuổi
entry_image_5 = PhotoImage(
    file=relative_to_assets("entry_5.png"))
entry_bg_5 = canvas.create_image(
    711.4999999999999,
    150.0,
    image=entry_image_5
)
entry_5 = Entry(
    cursor="arrow",
    state="disabled",
    bd=0,
    bg="#7990E3",
    fg="#000716",
    highlightthickness=0
)
entry_5.place(
    x=649.9999999999999,
    y=137.0,
    width=123.0,
    height=24.0
)

window.resizable(False, False)
window.mainloop()
