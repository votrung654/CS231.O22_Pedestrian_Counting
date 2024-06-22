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
Chạy file demo.py để xem kết quả. Sau khi chạy file demo.py sẽ xuất hiện 1 cửa sổ để ta upload ảnh. Upload ảnh và xem kết quả. 

## Kết quả
<img src="https://i.imgur.com/jx2IdeK.png">
<img src="https://i.imgur.com/CHTjeZD.png">

## Training
Code này train model sử dụng dataset [INRIA Person Dataset](http://pascal.inrialpes.fr/data/human/).

**Note**: Hoặc tải từ link sau: [link](https://drive.google.com/file/d/14GD_pBpBsprPiZlkmtXN_y5K72To16if/view?usp=sharing) nếu link trên không hoạt động.

Chạy đoạn command sau:
```
sudo sh fixpng.sh # Fix các file png bị lỗi trong dataset
```
**Note:** *Bước này là cần thiết để classifier được trained đúng cách*

Để chạy file train, run command sau:
```
python train.py --pos <path to positive images> --neg <path to negative images>
```
Ví dụ:
```
python train.py --pos INRIAPerson/train_64x128_H96/pos --neg INRIAPerson/train_64x128_H96/neg
```

Sau khi chạy xong, sẽ có 2 file  `person.pkl` and `person_final.pkl`, lần lượt là file huấn luyện lần 1 và huấn luyện lần 2 (sử dụng kỹ thuật hard negatively mined)

## Testing
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



