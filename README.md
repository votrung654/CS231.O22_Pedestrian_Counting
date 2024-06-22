<p align="center">
  <a href="https://www.uit.edu.vn/" title="Trường Đại học Công nghệ Thông tin" style="border: 5;">
    <img src="https://i.imgur.com/WmMnSRt.png" alt="Trường Đại học Công nghệ Thông tin | University of Information Technology">
  </a>
</p>

<!-- Title -->
<h1 align="center"><b>CS231.O22 - NHẬP MÔN THỊ GIÁC MÁY TÍNH</b></h1>



## MỤC LỤC
* [ Giới thiệu bản thân](#gioithieubanthan)
* [ Giới thiệu môn học](#gioithieumonhoc)
* [ Giảng viên hướng dẫn](#giangvien)
* [ Thành viên nhóm](#thanhvien)
* [ Đồ án môn học](#doan)

## GIỚI THIỆU MÔN HỌC
<a name="gioithieumonhoc"></a>
* **Tên môn học**: NHẬP MÔN THỊ GIÁC MÁY TÍNH - COMPUTER VISION
* **Mã môn học**: CS114
* **Mã lớp học**: CS114.O21
* **Năm học**: 2023-2024 (HK2), 19/02/2024 - 08/06/2024

## GIẢNG VIÊN HƯỚNG DẪN
<a name="giangvien"></a>
* PGS.TS. **Mai Tiến Dũng** - *dungmt@uit.edu.vn *

## THÀNH VIÊN NHÓM
<a name="thanhvien"></a>
| STT    | MSSV          | Họ và Tên              | Github                                               | Email                   |
| ------ |:-------------:| ----------------------:|-----------------------------------------------------:|-------------------------:
| 1      | 22520518      | Nguyễn Thanh Hùng      | https://github.com/nth4002                           |22520518@gm.uit.edu.vn   |
| 2      | 22521571      | Võ Đình Trung          | https://github.com/votrung654                        |22521571@gm.uit.edu.vn   |

## ĐỒ ÁN MÔN HỌC
<a name="doan"></a>
Đếm số người trong ảnh.

## Cách dùng source code trên
**Note**: Các file test và train chỉ sử dụng cho khả năng phân loại của mô hình, hiện nhóm chưa thực hiện việc đánh giá mô hình qua việc so sánh nhãn dự đoán và nhãn thực tế.
Chạy file demo.py để xem kết quả. Sau khi chạy file demo.py sẽ xuất hiện 1 cửa sổ để ta upload ảnh. Upload ảnh và xem kết quả. Dưới đây là một số câu lệnh để cài đặt các thư việc cần thiết nếu chúng chưa được cài đặt:
```
pip install opencv-python-headless
pip install numpy
pip install scikit-image
pip install joblib
pip install argparse
pip install pillow
pip install tkinter
```
Nếu không muốn sử dụng tkinter, có thể sử dụng câu lệnh sau trong terminal:
```
python demo.py -i/--image <path to image> 
```
## Kết quả
<img src="https://i.imgur.com/jx2IdeK.png">
<img src="https://i.imgur.com/CHTjeZD.png">
Ảnh demo được lưu trong thư mục demo_output dưới định dạng jpg, tên cụ thể là output_image.jpg. Chúng em đã đính kèm 2 ảnh cho thấy kết quả khi chạy trên câu lệnh có đường dẫn ảnh trên terminal hoặc dùng tkinter thông qua việc run một cách thông thường.
## Training
Code này sử dụng dataset [INRIA Person Dataset](http://pascal.inrialpes.fr/data/human/).

**Note**: Vì link chính thức không hoạt động vì một số lý do, đây là link thay thế: [link](https://drive.google.com/file/d/14GD_pBpBsprPiZlkmtXN_y5K72To16if/view?usp=sharing).

Để chạy file train_SVM.py, run command sau:
```
python train_SVM.py --pos <path to positive images> --neg <path to negative images>
```
Ví dụ:
```
python train_SVM.py --pos INRIAPerson/train_64x128_H96/pos --neg INRIAPerson/train_64x128_H96/neg
```

Sau khi chạy xong, sẽ có 2 file  `pre-hard_negative_mining_SVM_model.pkl` and `after-hard_negative_mining_SVM_model`, lần lượt là file huấn luyện lần 1 và huấn luyện lần 2 (sử dụng kỹ thuật hard negatively mined)

tương tự với file train_LR.py, run command sau:
```
python train_LR.py --pos <path to positive images> --neg <path to negative images>
```
Ví dụ:
```
python train_LR.py --pos INRIAPerson/train_64x128_H96/pos --neg INRIAPerson/train_64x128_H96/neg
```

Sau khi chạy xong, sẽ có 2 file  `pre-hard_negative_mining_LR_model.pkl` and `after-hard_negative_mining_LR_model`, lần lượt là file huấn luyện lần 1 và huấn luyện lần 2 (sử dụng kỹ thuật hard negatively mined)

## Testing
Trước khi sử dụng code, cần chọn mô hình qua dòng lệnh sau trong file test.py (ở đây được lấy ví dụ từ model được lưu trong after-hard_negative_mining_SVM_model.pkl):
```
clf = joblib.load('after-hard_negative_mining_SVM_model.pkl')
```
Chạy đoạn command sau để test:
```
python test.py --pos <path to positive images> --neg <path to negative images>
```
Ví dụ:
```
python test.py --pos INRIAPerson/test_64x128_H96/pos --neg INRIAPerson/test_64x128_H96/neg
```

Nó sẽ in ra `True Positives`, `True Negatives`, `False Positives`, `False Negatives`, `Precision`, `Recall` và `F1 Score`.

## Kết quả sau khi chạy 
`SVM model`

<img src="https://i.imgur.com/LjsiN9B.png">

`Logistic Regression model`

<img src="https://i.imgur.com/dtX7NHb.png">



