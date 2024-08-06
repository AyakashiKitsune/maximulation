import shutil
from time import sleep
from tkinter import *
from tkinter import filedialog
from tkinter import scrolledtext
import Levenshtein
from PIL import ImageTk, Image
import os
from pathlib import Path
import tensorflow as tf
from generator_filter import generate_iso_noise,generate_motion_blur,generate_randombrightnesscontrast,resize_image
from maxim_infer import deblur_infer as maxim_deblur_infer
from maxim_infer import denoise_infer as maxim_denoise_infer
from maxim_infer import enhancement_infer as maxim_enhancement_infer
from hinet_infer import deblur_infer as hinet_deblur_infer
from mprnet_infer import denoise_infer as mprnet_denoise_infer
from uegan_infer import enhancement_infer as uegan_enhancement_infer
from pytesseract import image_to_string

HOME_PATH = str(Path.home())
BASE_PATH = os.getcwd()
INPUT_IMAGE = os.path.join(BASE_PATH,"img", "input", "input.png")
ORIGINAL_DIR = os.path.join(BASE_PATH,"img", "original")
ORIGINAL_IMAGE =os.path.join(ORIGINAL_DIR,"input.png") #RESIZED

HINET_RESULT_DEBLUR = os.path.join(BASE_PATH,"img", "output", "deblur", "hinet","input.png")
MAXIM_RESULT_DEBLUR = os.path.join(BASE_PATH,"img", "output", "deblur", "maxim","input.png")

MPRNET_RESULT_DENOISE = os.path.join(BASE_PATH,"img", "output", "denoise", "mprnet", "input.png")
MAXIM_RESULT_DENOISE = os.path.join(BASE_PATH,"img", "output", "denoise", "maxim", "input.png")

UEGAN_RESULT_ENHANCEMENT = os.path.join(BASE_PATH,"img", "output", "enhancement", "uegan", "input.png")
MAXIM_RESULT_ENHANCEMENT = os.path.join(BASE_PATH,"img", "output", "enhancement", "maxim", "input.png")

states ={
    "deblur" : {
        "titlebar" : "deblur",
        "othermodel" : "HInet"
    },
    "denoise" : {
        "titlebar" : "denoise",
        "othermodel" : "MPRnet"
    },
    "enhancement" : {
        "titlebar" : "enhancement",
        "othermodel" : "UEGAN"
    },
}

titlebar = states["deblur"]

root = Tk()
root.title(titlebar["titlebar"])
root.geometry()
IMAGE_W,IMAGE_H = 450,600

input_image = ImageTk.PhotoImage(Image.open(os.path.join(ORIGINAL_DIR,"input.png")).resize((IMAGE_W, IMAGE_H)), )
filter_image = ImageTk.PhotoImage(Image.open(os.path.join(ORIGINAL_DIR,"input.png")).resize((IMAGE_W, IMAGE_H)),)
maxim_image = ImageTk.PhotoImage(Image.open(os.path.join(ORIGINAL_DIR,"input.png")).resize((IMAGE_W, IMAGE_H)),)
other_model_image = ImageTk.PhotoImage(Image.open(os.path.join(ORIGINAL_DIR,"input.png")).resize((IMAGE_W, IMAGE_H)),)
label_other_model_text =    Label(text="{} result".format(titlebar["othermodel"]))
title_rows = [
    Label(text="input image"),
    Label(text="filtered iamge"),
    Label(text="Maxim result"),
    label_other_model_text
]

label_input_image = Label(image=input_image)
label_filter_image = Label(image=filter_image)
label_maxim_image = Label(image=maxim_image)
label_other_model_image = Label(image=other_model_image)
image_rows = [
    label_input_image,
    label_filter_image,
    label_maxim_image,
    label_other_model_image
]
global ORIGINAL_OCR
ORIGINAL_OCR = "---"
global STATE
STATE  = "deblur"

def changeState_blur(_state = "deblur"):
    titlebar = states[_state]
    root.title(titlebar["titlebar"])
    global STATE
    STATE= _state
    
    label_other_model_text["text"] = "{} result".format(titlebar["othermodel"])

def changeState_noise(_state = "denoise") -> None:
    titlebar = states[_state]
    global STATE
    STATE= _state
    root.title(titlebar["titlebar"])
    label_other_model_text["text"] = "{} result".format(titlebar["othermodel"])


def changeState_enhance(_state = "enhancement"):
    titlebar = states[_state]
    global STATE
    STATE= _state
    root.title(titlebar["titlebar"])
    label_other_model_text["text"] = "{} result".format(titlebar["othermodel"])

deblur_tab =  Button(text="deblur",command=changeState_blur)
denoise_tab = Button(text="denoise",command=changeState_noise)
enhance_tab = Button(text="enhancement",command=changeState_enhance)

tabs = [
    deblur_tab,
    denoise_tab,
    enhance_tab
]
for index,tab in enumerate(tabs):
    tab.grid(row=0,column=index)


for index,title in enumerate(title_rows):
    title.grid(row=1, column=index, padx=4)

for index, image in enumerate(image_rows):
    image.grid(row=2,column=index)

def select_image():
    # os.path.join(BASE_PATH, "0.jpg")
    filename = filedialog.askopenfilename(initialdir=HOME_PATH, title="Select An Image", filetypes=(("jpeg files", "*.jpg"), ("png files", "*.png")))
    new_label = ImageTk.PhotoImage(Image.open(filename).resize((IMAGE_W, IMAGE_H)))

    label_input_image.config(image=new_label)
    label_input_image.image =new_label

    scrolltext_original_ocr.configure(state=NORMAL)
    scrolltext_original_ocr.delete(1.0,END)
    global ORIGINAL_OCR 
    ORIGINAL_OCR = read_image_ocr_image(filename)
    scrolltext_original_ocr.insert(INSERT, ORIGINAL_OCR)
    scrolltext_original_ocr.configure(state=DISABLED)


    shutil.copyfile(filename, os.path.join(ORIGINAL_DIR,"input.png"))
    resize_image(img=os.path.join(ORIGINAL_DIR,"input.png"), save=True,dist=os.path.join(ORIGINAL_DIR,"input.png"))

    if STATE == "deblur":
        generate_motion_blur(filename,INPUT_IMAGE)
    elif STATE == "denoise":
        generate_iso_noise(filename, INPUT_IMAGE)
    elif STATE == "enhancement":
        generate_randombrightnesscontrast(filename,INPUT_IMAGE)
    else:
        raise Exception("wrong task of model")
    
    scrolltext_filter_ocr.configure(state=NORMAL)
    scrolltext_filter_ocr.delete(1.0,END)
    filter_ocr = read_image_ocr_image(INPUT_IMAGE)
    scrolltext_filter_ocr.insert(INSERT,filter_ocr)
    scrolltext_filter_ocr.configure(state=DISABLED)
    label_filter_ocr["text"] = "original vs filter: {}".format(measure_levishtein_distance(filter_ocr, ORIGINAL_OCR))

    sleep(1)
    filterimage = ImageTk.PhotoImage(Image.open(INPUT_IMAGE).resize((IMAGE_W, IMAGE_H)))
    label_filter_image.config(image= filterimage)
    label_filter_image.image = filterimage

Button(text="select an image", command=select_image).grid(row=3,column=0)


def get_psnr( img1, img2):
    orig = tf.keras.utils.load_img(img1,)
    result = tf.keras.utils.load_img(img2)
    origa =tf.keras.utils.img_to_array(orig,dtype="uint8")
    resulta = tf.keras.utils.img_to_array(result,dtype="uint8")
    return tf.image.psnr(origa, resulta,255)

def get_ssim(img1, img2):
    orig = tf.keras.utils.load_img(img1)
    result = tf.keras.utils.load_img(img2)
    origa =tf.keras.utils.img_to_array(orig,dtype="uint8")
    resulta = tf.keras.utils.img_to_array(result,dtype="uint8")
    return tf.image.ssim(origa,resulta,255)

def infer():
    if STATE == "deblur":
        root.title("running debluring in maxim model")
        maxim_deblur_infer(None)
        maxim_image = ImageTk.PhotoImage(Image.open(MAXIM_RESULT_DEBLUR).resize((IMAGE_W, IMAGE_H)))
        label_maxim_image.config(image = maxim_image)
        label_maxim_image.image = maxim_image
        label_psnr_maxim["text"] = "PSNR: {}".format(get_psnr(MAXIM_RESULT_DEBLUR, ORIGINAL_IMAGE))
        label_ssim_maxim["text"] = "SSIM: {}".format(get_ssim(MAXIM_RESULT_DEBLUR, ORIGINAL_IMAGE))
        maxim_ocr = read_image_ocr_image(MAXIM_RESULT_DEBLUR)
        label_maxim_ocr["text"] = "original vs maxim: {}".format(measure_levishtein_distance(maxim_ocr, ORIGINAL_OCR))

        scrolltext_maxim_ocr.configure(state=NORMAL)
        scrolltext_maxim_ocr.delete(1.0,END)
        scrolltext_maxim_ocr.insert(INSERT, maxim_ocr)
        scrolltext_maxim_ocr.configure(state=DISABLED)


        root.title("running debluring in  HINet model")
        hinet_deblur_infer(None)
        other_model_image = ImageTk.PhotoImage(Image.open(HINET_RESULT_DEBLUR).resize((IMAGE_W, IMAGE_H)))
        label_other_model_image.config(image = other_model_image)
        label_other_model_image.image = other_model_image
        label_psnr_other["text"] =  "PSNR: {}".format(get_psnr(HINET_RESULT_DEBLUR, ORIGINAL_IMAGE))
        label_ssim_other["text"] = "SSIM: {}".format(get_ssim(HINET_RESULT_DEBLUR, ORIGINAL_IMAGE))
        other_model_ocr = read_image_ocr_image(HINET_RESULT_DEBLUR)
        label_other_model_ocr["text"] = "original vs other model: {}".format(measure_levishtein_distance(other_model_ocr, ORIGINAL_OCR))
        scrolltext_other_model_ocr.configure(state=NORMAL)
        scrolltext_other_model_ocr.delete(1.0,END)
        scrolltext_other_model_ocr.insert(INSERT, other_model_ocr)
        scrolltext_other_model_ocr.configure(state=DISABLED)

        root.title("done debluring")
    elif STATE == "denoise":
        root.title("running denoising in maxim model")
        maxim_denoise_infer(None)
        maxim_image = ImageTk.PhotoImage(Image.open(MAXIM_RESULT_DENOISE).resize((IMAGE_W, IMAGE_H)))
        label_maxim_image.config(image = maxim_image)
        label_maxim_image.image = maxim_image
        label_psnr_maxim["text"] = "PSNR: {}".format(get_psnr(MAXIM_RESULT_DENOISE, ORIGINAL_IMAGE))
        label_ssim_maxim["text"] = "SSIM: {}".format(get_ssim(MAXIM_RESULT_DENOISE, ORIGINAL_IMAGE))
        maxim_ocr = read_image_ocr_image(MAXIM_RESULT_DENOISE)
        label_maxim_ocr["text"] = "original vs maxim: {}".format(measure_levishtein_distance(maxim_ocr, ORIGINAL_OCR))
        scrolltext_maxim_ocr.configure(state=NORMAL)
        scrolltext_maxim_ocr.delete(1.0,END)
        scrolltext_maxim_ocr.insert(INSERT, maxim_ocr)
        scrolltext_maxim_ocr.configure(state=DISABLED)

        root.title("running denoising in mprnet model")
        mprnet_denoise_infer(None)
        other_model_image = ImageTk.PhotoImage(Image.open(MPRNET_RESULT_DENOISE).resize((IMAGE_W, IMAGE_H)))
        label_other_model_image.config(image = other_model_image)
        label_other_model_image.image = other_model_image
        label_psnr_other["text"] = "PSNR: {}".format(get_psnr(MPRNET_RESULT_DENOISE, ORIGINAL_IMAGE))
        label_ssim_other["text"] = "SSIM: {}".format(get_ssim(MPRNET_RESULT_DENOISE, ORIGINAL_IMAGE))
        other_model_ocr = read_image_ocr_image(MPRNET_RESULT_DENOISE)
        label_other_model_ocr["text"] = "original vs other model: {}".format(measure_levishtein_distance(other_model_ocr, ORIGINAL_OCR))
        scrolltext_other_model_ocr.configure(state=NORMAL)
        scrolltext_other_model_ocr.delete(1.0,END)
        scrolltext_other_model_ocr.insert(INSERT,other_model_ocr )
        scrolltext_other_model_ocr.configure(state=DISABLED)

        root.title("done denoising")

    elif STATE == "enhancement":
        root.title("running enhancement in maxim model")
        maxim_enhancement_infer(None)
        maxim_image = ImageTk.PhotoImage(Image.open(MAXIM_RESULT_ENHANCEMENT).resize((IMAGE_W, IMAGE_H)))
        label_maxim_image.config(image = maxim_image)
        label_maxim_image.image = maxim_image
        label_psnr_maxim["text"] = "PSNR: {}".format(get_psnr(MAXIM_RESULT_ENHANCEMENT, ORIGINAL_IMAGE))
        label_ssim_maxim["text"] = "SSIM: {}".format(get_ssim(MAXIM_RESULT_ENHANCEMENT, ORIGINAL_IMAGE))
        maxim_ocr = read_image_ocr_image(MAXIM_RESULT_ENHANCEMENT)
        label_maxim_ocr["text"] = "original vs maxim: {}".format(measure_levishtein_distance(maxim_ocr, ORIGINAL_OCR))
        scrolltext_maxim_ocr.configure(state=NORMAL)
        scrolltext_maxim_ocr.delete(1.0,END)
        scrolltext_maxim_ocr.insert(INSERT,maxim_ocr )
        scrolltext_maxim_ocr.configure(state=DISABLED)

        root.title("running enhancement in uegan model")
        uegan_enhancement_infer(None)
        other_model_image = ImageTk.PhotoImage(Image.open(UEGAN_RESULT_ENHANCEMENT).resize((IMAGE_W, IMAGE_H)))
        label_other_model_image.config(image = other_model_image)
        label_other_model_image.image = other_model_image
        label_psnr_other["text"] = "PSNR: {}".format(get_psnr(UEGAN_RESULT_ENHANCEMENT, ORIGINAL_IMAGE))
        label_ssim_other["text"] = "SSIM: {}".format(get_ssim(UEGAN_RESULT_ENHANCEMENT, ORIGINAL_IMAGE))
        other_model_ocr = read_image_ocr_image(UEGAN_RESULT_ENHANCEMENT)
        label_other_model_ocr["text"] = "original vs other model: {}".format(measure_levishtein_distance(other_model_ocr, ORIGINAL_OCR))
        scrolltext_other_model_ocr.configure(state=NORMAL)
        scrolltext_other_model_ocr.delete(1.0,END)
        scrolltext_other_model_ocr.insert(INSERT, other_model_ocr)
        scrolltext_other_model_ocr.configure(state=DISABLED)
        root.title("done enhancement")

Button(text="inference", command=infer).grid(row=3,column=1)

label_psnr_maxim = Label(text="PSNR: ")
label_psnr_maxim.grid(row=3,column=2)
label_ssim_maxim = Label(text="SSIM: ")
label_ssim_maxim.grid(row=4,column=2)
label_psnr_other = Label(text="PSNR: ")
label_psnr_other.grid(row=3,column=3)
label_ssim_other = Label(text="SSIM: ")
label_ssim_other.grid(row=4,column=3)
Label(text="PSNR: tells how noisy the image").grid(row=5,column=2)
Label(text="SSMIN: closer to 1.0 the better").grid(row=5,column=3)




def measure_levishtein_distance(text1, text2):
    return Levenshtein.ratio(text1, text2)

def read_image_ocr_image(path):
    image = Image.open(path)
    # text = tesserocr.image_to_text(image).strip()
    text = image_to_string(image).strip()
    return text

Label(text="original ocr").grid(row=6,column=0)
label_original_ocr = Label(text="")
label_original_ocr.grid(row=7,column=0)
scrolltext_original_ocr = scrolledtext.ScrolledText(root, width=50, height=10, wrap=WORD)
scrolltext_original_ocr.grid(row=8,column=0)
scrolltext_original_ocr.insert(INSERT,', '.join(["aldkajslkd" for i in range(100)]),)
scrolltext_original_ocr.configure(state=DISABLED)


Label(text="filter ocr").grid(row=6,column=1)
label_filter_ocr = Label(text="")
label_filter_ocr.grid(row=7,column=1)
scrolltext_filter_ocr = scrolledtext.ScrolledText(root, width=50, height=10, wrap=WORD)
scrolltext_filter_ocr.grid(row=8,column=1)
scrolltext_filter_ocr.insert(INSERT,', '.join(["aldkajslkd" for i in range(100)]),)
scrolltext_filter_ocr.configure(state=DISABLED)

Label(text="maxim ocr").grid(row=6,column=2)
label_maxim_ocr = Label(text="")
label_maxim_ocr.grid(row=7,column=2)
scrolltext_maxim_ocr = scrolledtext.ScrolledText(root, width=50, height=10, wrap=WORD)
scrolltext_maxim_ocr.grid(row=8,column=2)
scrolltext_maxim_ocr.insert(INSERT,', '.join(["aldkajslkd" for i in range(100)]),)
scrolltext_maxim_ocr.configure(state=DISABLED)

Label(text="other model ocr").grid(row=6,column=3)
label_other_model_ocr = Label(text="")
label_other_model_ocr.grid(row=7,column=3)
scrolltext_other_model_ocr = scrolledtext.ScrolledText(root, width=50, height=10, wrap=WORD)
scrolltext_other_model_ocr.grid(row=8,column=3)
scrolltext_other_model_ocr.insert(INSERT,', '.join(["aldkajslkd" for i in range(100)]),)
scrolltext_other_model_ocr.configure(state=DISABLED)







root.mainloop()
