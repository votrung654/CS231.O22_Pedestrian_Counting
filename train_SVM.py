import cv2
from sklearn import svm
import os
import numpy as np
import joblib
from skimage.feature import hog
from sklearn.utils import shuffle
import sys
import argparse
import random

MAX_HARD_NEGATIVES = 20000

# Phân tích cú pháp thư mục train
parser = argparse.ArgumentParser(description='Phân tích cú pháp để truy cập thư mục train')
parser.add_argument('--pos', help='Đường dẫn đến thư mục chứa ảnh có người')
parser.add_argument('--neg', help='Đường dẫn đến thư mục chứa ảnh không có người')

args = parser.parse_args()
pos_img_dir = args.pos
neg_img_dir = args.neg

# Cắt ảnh ở trung tâm
def crop_centre(img):
    h, w, _ = img.shape
    l = (w - 64)//2
    t = (h - 128)//2

    crop = img[t:t+128, l:l+64]
    return crop

# Chọn ngẫu nhiên 10 cửa sổ
def ten_random_windows(img):
    h, w = img.shape
    if h < 128 or w < 64:
        return []

    h = h - 128
    w = w - 64

    windows = []

    for i in range(0, 10):
        x = random.randint(0, w)
        y = random.randint(0, h)
        windows.append(img[y:y+128, x:x+64])

    return windows

# Đọc tên file
def read_filenames():

    f_pos = []
    f_neg = []

    mypath_pos = pos_img_dir
    for (dirpath, dirnames, filenames) in os.walk(mypath_pos):
        f_pos.extend(filenames)
        break

    mypath_neg = neg_img_dir
    for (dirpath, dirnames, filenames) in os.walk(mypath_neg):
        f_neg.extend(filenames)
        break

    return f_pos, f_neg

# Đọc ảnh
def read_images(pos_files, neg_files):

    X = []
    Y = []

    pos_count = 0

    for img_file in pos_files:
        print(os.path.join(pos_img_dir, img_file))
        img = cv2.imread(os.path.join(pos_img_dir, img_file))

        cropped = crop_centre(img)

        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        features = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2", transform_sqrt=True, feature_vector=True)
        pos_count += 1

        X.append(features)
        Y.append(1)


    neg_count = 0

    for img_file in neg_files:
        print(os.path.join(neg_img_dir, img_file))
        img = cv2.imread(os.path.join(neg_img_dir, img_file))
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        windows = ten_random_windows(gray_img)

        for win in windows:
            features = hog(win, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2", transform_sqrt=True, feature_vector=True)
            neg_count += 1
            X.append(features)
            Y.append(0)


    return X, Y, pos_count, neg_count

# Cửa sổ trượt
def sliding_window(image, window_size, step_size):
    '''
    Hàm này trả về một phần của ảnh đầu vào `image` có kích thước bằng với `window_size`. Ảnh đầu tiên trả về có tọa độ trên cùng bên trái là (0, 0)
    và được tăng lên theo cả hướng x và y bởi `step_size` được cung cấp.
    Vì vậy, các tham số đầu vào là -
    * `image` - Ảnh đầu vào
    * `window_size` - Kích thước của cửa sổ trượt
    * `step_size` - Kích thước tăng của cửa sổ

    Hàm trả về một tuple -
    (x, y, im_window)
    nơi
    * x là tọa độ x trên cùng bên trái
    * y là tọa độ y trên cùng bên trái
    * im_window là ảnh của cửa sổ trượt
    '''
    for y in range(0, image.shape[0]-128, step_size[1]):
        for x in range(0, image.shape[1]-64, step_size[0]):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

# Thực hiện hard negative mining
def hard_negative_mine(f_neg, winSize, winStride):

    hard_negatives = []
    hard_negative_labels = []

    count = 0
    num = 0
    for imgfile in f_neg:

        img = cv2.imread(os.path.join(neg_img_dir, imgfile))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for (x, y, im_window) in sliding_window(gray, winSize, winStride):
            features = hog(im_window, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2", transform_sqrt=True, feature_vector=True)
            if (clf1.predict([features]) == 1):
                hard_negatives.append(features)
                hard_negative_labels.append(0)

                count = count + 1

            if (count == MAX_HARD_NEGATIVES):
                return np.array(hard_negatives), np.array(hard_negative_labels)

        num = num + 1

        sys.stdout.write("\r" + "\tHard negative mining: " + str(count) + "\tHoàn thành: " + str(round((count / float(MAX_HARD_NEGATIVES))*100, 4)) + " %" )

        sys.stdout.flush()

    return np.array(hard_negatives), np.array(hard_negative_labels)



pos_img_files, neg_img_files = read_filenames()

print("Tổng số ảnh có người: " + str(len(pos_img_files)))
print("Tổng số không có người: " + str(len(neg_img_files)))
print("Đang đọc Ảnh")

X, Y, pos_count, neg_count = read_images(pos_img_files, neg_img_files)

X = np.array(X)
Y = np.array(Y)

X, Y = shuffle(X, Y, random_state=0)


print("Ảnh đã được đọc và xáo trộn")
print("Có người: " + str(pos_count))
print("Không có người: " + str(neg_count))
print("Bắt đầu train")

clf1 = svm.LinearSVC(C=0.01, max_iter=1000, class_weight='balanced', verbose = 1)


clf1.fit(X, Y)
print("Đã train xong")


joblib.dump(clf1, 'pre-hard_negative_mining_SVM_model.pkl')


print("Thực hiện hard negative mining")

winStride = (4, 4)
winSize = (64, 128)

print("Số lượng mẫu hard negative mining: " + str(MAX_HARD_NEGATIVES))

hard_negatives, hard_negative_labels = hard_negative_mine(neg_img_files, winSize, winStride)

sys.stdout.write("\n")

hard_negatives = np.concatenate((hard_negatives, X), axis = 0)
hard_negative_labels = np.concatenate((hard_negative_labels, Y), axis = 0)

hard_negatives, hard_negative_labels = shuffle(hard_negatives, hard_negative_labels, random_state=0)

print("Kích thước mẫu sau cùng: " + str(hard_negatives.shape))
print("Train lại bộ phân loại với dữ liệu sau cùng")

clf2 = svm.LinearSVC(C=0.01, max_iter=1000, class_weight='balanced', verbose = 1)

clf2.fit(hard_negatives, hard_negative_labels)

print("Đã train và đang lưu")

joblib.dump(clf2, 'after-hard_negative_mining_SVM_model.pkl')