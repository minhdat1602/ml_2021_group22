# Link dataset từ kaggle
# https://www.kaggle.com/aneerbanchakraborty/face-mask-detection-data?select=with_mask
# 1915 hình ảnh có khẩu trang.
# 1918 hình ảnh không khẩu trang

# Import các thư viện
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.svm import SVC

# Khởi tạo hằng số
LEARNING_RATE = 0.0001
BATH_SIZE = 32
EPOCHS = 20
IMAGE_SIZE = 224

CATEGORIES = ["with_mask", "without_mask"]
DIRECTORY = r"E:\NAM3_HOCKY2\MACHINE_LEARNING\phathienkhautrang\dataset"

# STEP1: TIỀN XỬ LÝ DỮ LIỆU
# Khởi tạo danh sách hình ảnh
data = []
# Khởi tạo danh sách nhãn của các hình ảnh (with_mask, without_mask)
labels = []

# Duyệt qua danh sách folder trong dataset.
for CATEGORY in CATEGORIES:
    path = os.path.join(DIRECTORY, CATEGORY)
    print(path, "có", len(os.listdir(path)), "hình ảnh")

    # Duyệt qua danh sách hình ảnh trong folder
    for image_name in os.listdir(path):
        image_path = os.path.join(path, image_name)
        img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        img = img_to_array(img)
        img = preprocess_input(img)  # xử lý dữ liệu phù hợp với mobileNetV2.

        data.append(img)
        labels.append(CATEGORY)

# Chuyển labels về 0, 1. (0: with_mask & 1: without_mask)
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# Chuyển list thành array
data = np.array(data, dtype="float32")
labels = np.array(labels)

# split data, 20% dùng để test, 80% dùng để train
# tratify = labels -> dữ liệu chia ra phù hợp với % nhãn 0, % nhãn 1 trong labels ban đầu.
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)

# Chuẩn hóa dữ liệu.
# X_train, X_test = X_train / 255.0, X_test / 255.0

# One-hot encoding
y_train_one_hot, y_test_one_hot = to_categorical(y_train), to_categorical(y_test)

# Tăng cường dữ liệu cho hình ảnh, tức generate ra các dữ liệu hình ảnh
# mà hình ảnh đó di chuyển sang trái-phải, trên-dưới, zoom, xoay...để tăng thêm độ đa dạng dữ liệu.
# Dùng hiệu quả cho mô hình mà có ít dữ liệu.
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

# STEP2: TRAINING.......................

# Khởi tạo base model (feature learning) cho model
# Dùng để học các đặc trưng của hình ảnh (Trích xuất đặt trưng).
# kiến trúc: MobileNetV2.
feature_extractor = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
# base_model = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# weights của các layer trong base model sẽ không được cập nhật trong quá trình training
for layer in feature_extractor.layers:
    layer.trainable = False

# Khởi tạo head model, fully connection
prediction_layer = feature_extractor.output
prediction_layer = AveragePooling2D(pool_size=(7, 7))(prediction_layer)
prediction_layer = Flatten(name="flatten")(prediction_layer)
prediction_layer = Dense(128, activation="relu")(prediction_layer)
prediction_layer = Dropout(0.5)(prediction_layer)
prediction_layer = Dense(2, activation="softmax")(prediction_layer)

# Khởi tạo CNN model
cnn_model = Model(inputs=feature_extractor.input, outputs=prediction_layer)

# Compile CNN model
print("[INFO] Compiling CNN model...")
optimizer = Adam(lr=LEARNING_RATE, decay=LEARNING_RATE / EPOCHS)
cnn_model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
# print(cnn_model.summary())

# Huấn luện cho model.
print("[INFO] Training the CNN Model...")
H = cnn_model.fit(aug.flow(X_train, y_train_one_hot, batch_size=BATH_SIZE),
                  steps_per_epoch=len(X_train) // BATH_SIZE,
                  validation_data=(X_test, y_test_one_hot),
                  validation_steps=len(X_test) // BATH_SIZE,
                  epochs=EPOCHS)

# Dự đoán tập test bằng mô hình CNN đã được huấn luyện.
prediction_CNN = cnn_model.predict(X_test, batch_size=BATH_SIZE)
prediction_CNN = np.argmax(prediction_CNN, axis=1)

print("CNN accuracy:", accuracy_score(y_test, prediction_CNN))
print(classification_report(y_test_one_hot.argmax(axis=1), prediction_CNN, target_names=lb.classes_))
print(cnn_model.evaluate(X_test, y_test_one_hot))

# RANDOM FOREST, SVM
# Sử dụng Random Forest để phân loại từ các đặc trưng.
features = feature_extractor.predict(X_train)
print("feature_extractor:", features.shape)
features = features.reshape(features.shape[0], -1)
X_for_RF = features
print("Reshape:", X_for_RF.shape)

# Khởi tạo Random Forest, SVM model.
RF_model = RandomForestClassifier(n_estimators=50, random_state=42)
SVM_model = SVC(kernel="linear", gamma="auto", C=1)

# Training Random Forest model
print("[INFO] Training the Random Forest model...")
RF_model.fit(X_for_RF, np.ravel(y_train))
print("[INFO] Training the SVM model...")
SVM_model.fit(X_for_RF, np.ravel(y_train))

# Dữ liệu tập Test đã thông qua quá trình Trích xuất.
X_test_feature = feature_extractor.predict(X_test)
X_test_features = X_test_feature.reshape(X_test_feature.shape[0], -1)

# Dự đoán tập test bằng RF, SVM đã được huấn luyện.
prediction_RF = RF_model.predict(X_test_features)
prediction_SVM = SVM_model.predict(X_test_features)

# Độ chính xác của mô hình RF, CNN.
print("RANDOM FOREST accuracy:", accuracy_score(y_test, prediction_RF))
print("SVM accuracy:", accuracy_score(y_test, prediction_SVM))

print("[INFO] SAVING MODEL...")
cnn_model.save("models/cnn.model", save_format="h5")
joblib.dump(RF_model, "models/random_forest.joblib")
joblib.dump(RF_model, "models/svm.joblib")
print("[INFO] -----------------END------------------")

acc = [round(accuracy_score(y_test, prediction_CNN)*100, 2), round(accuracy_score(y_test, prediction_RF)*100, 2),
       round(accuracy_score(y_test, prediction_SVM)*100, 2)]
tt = ['CNN', "Random Forest", "SVM"]
plt.bar(tt, acc, color="blue")
plt.title("Độ chính xác của mô hình")
plt.xlabel("Mô hình")
plt.ylabel("Độ chính xác")
for index, data in enumerate(acc):
    plt.text(x=index, y=data+1, s=f"{data}", fontdict=dict(fontsize=20))
plt.show()
