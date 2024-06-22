import cv2
from sklearn import svm
import os
import numpy as np
import joblib
from skimage.feature import hog
import argparse

parser = argparse.ArgumentParser(description='Phân tích cú pháp để truy cập thư mục train')
parser.add_argument('--pos', help='Đường dẫn đến thư mục chứa ảnh có người')
parser.add_argument('--neg', help='Đường dẫn đến thư mục chứa ảnh không có người')

args = parser.parse_args()

pos_img_dir = args.pos
neg_img_dir = args.neg
#Ví dụ sử dụng để test model SVM (có thể thay bằng model Logistic Regression quan file after-hard_negative_mining_LR_model.pkl)
#Cũng có thể sử dụng model trước khi hard negative mining bằng file pre-hard_negative_mining_SVM_model.pkl
clf = joblib.load('after-hard_negative_mining_SVM_model.pkl')

total_pos_samples = 0
total_neg_samples = 0

def crop_centre(img):
    h, w, d = img.shape
    # Kiểm tra nếu hình ảnh quá nhỏ để cắt
    if h < 128 or w < 64:
        print(f"Hình ảnh quá nhỏ để cắt: {h}x{w}")
        return None
    l = (w - 64)//2
    t = (h - 128)//2
    # Cắt hình ảnh từ trung tâm
    crop = img[t:t+128, l:l+64]
    return crop

def read_filenames():
    f_pos = []
    f_neg = []

    # Đọc tên file từ thư mục chứa hình ảnh có người
    for (dirpath, dirnames, filenames) in os.walk(pos_img_dir):
        f_pos.extend(filenames)
        break

    # Đọc tên file từ thư mục chứa hình ảnh không có người
    for (dirpath, dirnames, filenames) in os.walk(neg_img_dir):
        f_neg.extend(filenames)
        break

    print("Số lượng mẫu hình ảnh có người: " + str(len(f_pos)))
    print("Số lượng mẫu hình ảnh không có người: " + str(len(f_neg)))

    return f_pos, f_neg

def read_images(f_pos, f_neg):

    print ("Đang đọc hình ảnh...")

    array_pos_features = []
    array_neg_features = []
    global total_pos_samples
    global total_neg_samples
    for imgfile in f_pos:
        img = cv2.imread(os.path.join(pos_img_dir, imgfile))
        cropped = crop_centre(img)
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        features = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2", feature_vector=True)
        array_pos_features.append(features.tolist())

        total_pos_samples += 1

    for imgfile in f_neg:
        img = cv2.imread(os.path.join(neg_img_dir, imgfile))
        cropped = crop_centre(img)
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        features = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2", feature_vector=True)
        array_neg_features.append(features.tolist())
        total_neg_samples += 1

    return array_pos_features, array_neg_features



pos_img_files, neg_img_files = read_filenames()

pos_features, neg_features = read_images(pos_img_files, neg_img_files)

pos_result = clf.predict(pos_features)
neg_result = clf.predict(neg_features)

true_positives = cv2.countNonZero(pos_result)
false_negatives = pos_result.shape[0] - true_positives

false_positives = cv2.countNonZero(neg_result)
true_negatives = neg_result.shape[0] - false_positives

print ("True Positives: " + str(true_positives), "False Positives: " + str(false_positives))
print ("True Negatives: " + str(true_negatives), "False Negatives: " + str(false_negatives))

precision = float(true_positives) / (true_positives + false_positives)
recall = float(true_positives) / (true_positives + false_negatives)
accuracy = float(true_positives + true_negatives) / (true_positives + false_positives + true_negatives + false_negatives)
f1 = 2*precision*recall / (precision + recall)

print ("Accuracy: " + str(accuracy))
print ("Precision: " + str(precision))
print("Recall: " + str(recall))
print ("F1 Score: " + str(f1))
