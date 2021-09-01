# Link dataset từ kaggle
# https://www.kaggle.com/aneerbanchakraborty/face-mask-detection-data?select=with_mask
# 1915 hình ảnh có khẩu trang.
# 1918 hình ảnh không khẩu trang

# Import thư viện
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.applications import VGG16
# from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import normalize
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import cv2
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Khởi tạo hằng số
INIT_LR = 0.0001
BS = 32
EPOCHS = 20
IMG_SIZE = 224

CATEGORIES = ["with_mask", "without_mask"]
DIRECTORY = r"E:\NAM3_HOCKY2\MACHINE_LEARNING\phathienkhautrang\dataset"

# Tiền xử lí
data = []
labels = []

for CATEGORY in CATEGORIES:
    path = os.path.join(DIRECTORY, CATEGORY)
    print(path, "có", len(os.listdir(path)), "hình ảnh")

    for image_name in os.listdir(path):
        image_path = os.path.join(path, image_name)
        try:
            img_arr = cv2.imread(image_path)
            img = cv2.resize(img_arr, (224, 224))

            data.append(img)
            labels.append(CATEGORY)
        except Exception as e:
            pass

#
# data = normalize(data, axis=1)
data=np.array(data).reshape(len(data), -1)

# Chuyển labels về 0 1, 0: with_mask & 1: without_mask
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
# One-hot Encoding
# labels = to_categorical(labels)

# Để dụng được numpy, dùng np.array
data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)

print(data.shape)
print(labels.shape)



# split data, 20% dùng để test, 80% dùng để train
# tratify = labels -> dữ liệu chia ra phù hợp với % nhãn 0, % nhãn 1 trong labels
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels)

# Tăng cường dữ liệu cho hình ảnh, tức generate ra các dữ liệu hình ảnh
# mà hình ảnh đó di chuyển sang trái-phải, trên-dưới, zoom, xoay...
# để tăng thêm độ đa dạng dữ liệu.

# aug = ImageDataGenerator(
#     rotation_range=20,
#     zoom_range=0.15,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.15,
#     horizontal_flip=True,
#     fill_mode="nearest")
print("11111")
model = SVC(kernel="rbf", C=1)
model.fit(X_train, np.ravel(y_train, order="C"))

print("22222")
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(accuracy)




# # Khởi tạo base model cho model
# # kiến trúc: MobileNetV2, Thời gian chạy: 35 phút, Độ chính xác: 99.22%
# base_model = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
#
# # kiến trúc: VGG16, chạy hơn 2 tiếng @@.
# # base_model = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
#
# # Khởi tạo head model, fully connection
# head_model = base_model.output
# head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
# head_model = Flatten(name="flatten")(head_model)
# head_model = Dense(128, activation="relu")(head_model)
# head_model = Dropout(0.5)(head_model)
# head_model = Dense(2, activation="softmax")(head_model)
#
# # Khởi tạo model
# model = Model(inputs=base_model.input, outputs=head_model)
#
# # weights của các layer trong base model sẽ không được cập nhật trong quá trình training
# #
# for layer in base_model.layers:
#     layer.trainable = False
#
# # compile model
# print("[INFO] compiling model...")
# optimizer = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
# model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
#
# # fit model
# print("[INFO] training...")
# H = model.fit(aug.flow(X_train, y_train, batch_size=BS),
#               steps_per_epoch=len(X_train) // BS,
#               validation_data=(X_test, y_test),
#               validation_steps=len(X_test) // BS,
#               epochs=EPOCHS)
#
# # dự đoán tập test
# print("[INFO] Đánh giá...")
# predictions = model.predict(X_test, batch_size=BS)
#
# # Mỗi hình ảnh trả về kết quả ví dụ như [0.8, 0.2]
# # vậy kết quả là 0 (max của arr)
# # => có mang khẩu trang
#
# # kết quả dự đoán
# predictions = np.argmax(predictions, axis=1)
#
# # report
# print(classification_report(y_test.argmax(axis=1), predictions, target_names=lb.classes_))
# print(model.evaluate(X_test, y_test))
#
# # save model
# print("[INFO] Lưu model...")
# # model.save("mask_detector.model", save_format="h5")
