
# import thư viện
import os
import numpy as np
import imutils
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from joblib import load
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input

# load faceNet model từ openCV
# The .prototxt file(s) which define the model architecture (i.e., the layers themselves)
# The .caffemodel file which contains the weights for the actual layers
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Load model phát hiện khẩu trang
#CNN
maskNet = load_model("models/cnn.model")
#SVM
# maskNet = load("models/svm.joblib")
# feature_extractor = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# function phát hiện khuôn mặt từ faceNet
# và dự đoán bằng maskNet
def detect_and_predict_mask(frame, faceNet, maskNet):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104, 117, 123))

    # phát hiện khuôn mặt
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # Khởi tạo list các khuôn mặt, vị trí và dự đoán tương ứng
    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        # Xác xuất của khuôn mặt
        confidence = detections[0, 0, i, 2]

        # Nếu khuôn mặt dự đoán là đúng, xác định vị trí của khuôn mặt đó
        if confidence > 0.5:

            # tính toán giới hạn của khuôn mặt
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            startX, startY, endX, endY = box.astype("int")

            # Đảm bảo khuôn mặt nằm trong frame.
            startX, startY = (max(0, startX), max(0, startY))
            endX, endY = (min(w - 1, endX), min(h - 1, endY))

            # Tiền xử lý khuôn mặt
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # Thêm vào danh sách
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # Tiến hành dự đoán các khuôn mặt.
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        # CNN
        preds = maskNet.predict(faces, batch_size=32)
        # SVM
        # X_test_feature = feature_extractor.predict(faces)
        # X_test_features = X_test_feature.reshape(X_test_feature.shape[0], -1)
        # prediction_SVM = maskNet.predict(X_test_features)
        # preds = prediction_SVM
    # trả về vị trí và dự đoán của các khuôn mặt tương ứng.
    return (locs, preds)


# Khởi tạo video capture
print("[INFO] Bất đầu ghi.....")
video_path = r"GaiXinhTruongDHNganHang.mp4"
cap = cv2.VideoCapture(video_path)
# cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, height=600)

    # locs: vị trí các khuôn mặt được phát hiện
    # preds: kết quả dự đoán các khuôn mặt
    locs, preds = detect_and_predict_mask(frame, faceNet, maskNet)

    # Thực hiện vẽ nhãn lên video
    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        #CNN
        (mask, withoutMask) = pred
        #SVM
        # mask, withoutMask = 0, 0
        # if pred == 0:
        #     mask = 1
        # if pred == 1:
        #     withoutMask = 1

        # Kết quả và màu sắc để hiển thị lên hình ảnh
        label = "Co khau trang" if mask > withoutMask else "Khong khau trang"
        color = (0, 255, 0) if mask > withoutMask else (0, 0, 255)

        # Xác xuất của nhãn
        #CNN, SVM -> comment
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # Hiển thị kết quả, xác xuất lên ảnh.
        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        # Hiển thị đánh dấu vị trí khuôn mặt.
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # Hiển thị video
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # Nhấn q để kết thúc video
    if key == ord("q"):
        break

# When everything done, release
# the video capture object
cap.release()
# Closes all the frames
cv2.destroyAllWindows()

