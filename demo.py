import cv2
import numpy as np
import time
from skimage.feature import hog
import joblib
from nms import nms
import argparse
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import os
output_dir = ".\\demo_output"

def appendRects(i, j, conf, c, rects,scaleFactor):
    x = int((j)*pow(scaleFactor, c))
    y = int((i)*pow(scaleFactor, c))
    w = int((64)*pow(scaleFactor, c))
    h = int((128)*pow(scaleFactor, c))
    rects.append((x, y, conf, w, h))

def upload_image():
    file_path = filedialog.askopenfilename()
    process_image(file_path, args["downscale"], args["winstride"], args["nms_threshold"], use_gui=True)

def process_image(file_path, downscale, winstride, nms_threshold, use_gui=False):
    orig = cv2.imread(file_path)
    img = orig.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scaleFactor = downscale
    inverse = 1.0 / scaleFactor
    winStride = (winstride, winstride)
    winSize = (128, 64)
    rects = []
    h, w = gray.shape
    count = 0
    total_windows = int((h - 128) / winStride[1]) * int((w - 64) / winStride[0])
    processed_windows = 0
    while h >= 128 and w >= 64:
        h, w = gray.shape
        horiz = w - 64
        vert = h - 128
        i = 0
        while i < vert:
            j = 0
            while j < horiz:
                portion = gray[i:i + winSize[0], j:j + winSize[1]]
                features = hog(portion, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2")
                result = clf.predict([features])
                if int(result[0]) == 1:
                    confidence = clf.decision_function([features])
                    appendRects(i, j, confidence, count, rects, scaleFactor)
                j += winStride[0]
                processed_windows += 1
                if use_gui:
                    progress['value'] = (processed_windows / total_windows) * 100
                    root.update_idletasks()
            i += winStride[1]
        gray = cv2.resize(gray, (int(w * inverse), int(h * inverse)), interpolation=cv2.INTER_AREA)
        count += 1
    nms_rects = nms(rects, nms_threshold)
    num_people = len(nms_rects)
    if use_gui:
        result_text.set(f"Số người phát hiện được: {num_people}")
    for (a, b, conf, c, d) in nms_rects:
        cv2.rectangle(img, (a, b), (a + c, b + d), (0, 255, 0), 2)
    # Xử lí ảnh trước khi lưu
    # Convert ảnh từ BGR sang RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if use_gui:
        display_image = Image.fromarray(img)
        photo = ImageTk.PhotoImage(display_image)
        label = tk.Label(image=photo)
        label.image = photo
        label.pack()
    # Convert ảnh sang BGR trước khi lưu
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # Lưu ảnh
    output_path = os.path.join(output_dir, "output_image.jpg")
    cv2.imwrite(output_path, img)
    print("Số người phát hiện được: ", num_people)

parser = argparse.ArgumentParser(description='Điều chỉnh các thông số')

parser.add_argument('-i', "--image", help="Path to the test image", required=False)
parser.add_argument('-d','--downscale', default=1.2, type=float)
parser.add_argument('-w', '--winstride', default=4, type=int)
parser.add_argument('-n', '--nms_threshold', default=0.05, type=float)
args = vars(parser.parse_args())

#Sử dụng model Logistic Regression sau hard negative mining để ví dụ
clf = joblib.load("after-hard_negative_mining_LR_model.pkl")

if args["image"]:
    process_image(args["image"], args["downscale"], args["winstride"], args["nms_threshold"], use_gui=False)
else:
    # Nếu không có đường dẫn ảnh được cung cấp, mở giao diện để upload ảnh
    root = tk.Tk()

    upload_button = tk.Button(root, text="Upload Image", command=upload_image)
    upload_button.pack()

    result_text = tk.StringVar()
    result_label = tk.Label(root, textvariable=result_text)
    result_label.pack()

    progress = ttk.Progressbar(root, orient='horizontal', length=300, mode='determinate')
    progress.pack()

    root.mainloop()