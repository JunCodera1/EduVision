import cv2
import numpy as np
import tensorflow as tf

# ==========================================
# CẤU HÌNH
# ==========================================
MODEL_PATH = 'Student Engagement Model.h5'

# GIẢ ĐỊNH 6 NHÃN (Bạn hãy thay đổi thứ tự này nếu thấy nhận diện sai)
# Thông thường các model Engagement chia làm các mức độ cảm xúc/tập trung
LABELS = [
    "Engaged",       # 0: Tập trung
    "Confused",      # 1: Bối rối / Không hiểu
    "Bored",         # 2: Chán nản
    "Drowsy",        # 3: Buồn ngủ
    "Frustrated",    # 4: Cáu gắt / Khó chịu
    "Looking Away"   # 5: Không nhìn vào màn hình
]

def main():
    # 1. Tải Model
    print(f"Dang tai model: {MODEL_PATH}...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Tai model thanh cong!")
    except Exception as e:
        print(f"Loi tai model: {e}")
        return

    # 2. Khoi dong Webcam
    # Thuong la 0, neu co nhieu cam thi thu 1, 2...
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Khong the mo Webcam.")
        return

    print("\nBAT DAU NHAN DIEN...")
    print("Nhan 'q' de thoat chuong trinh.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Khong nhan duoc tin hieu tu Camera.")
            break

        # 3. Tien xu ly anh (Preprocessing)
        # Model yeu cau (256, 256, 3)
        # Resize anh
        resized_frame = cv2.resize(frame, (256, 256))
        
        # Chuyen he mau BGR (OpenCV) sang RGB (Keras/TensorFlow)
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        
        # Chuan hoa (Normalize) ve khoang 0-1
        normalized_frame = rgb_frame.astype('float32') / 255.0
        
        # Them chieu Batch: (256, 256, 3) -> (1, 256, 256, 3)
        input_data = np.expand_dims(normalized_frame, axis=0)

        # 4. Du doan (Predict)
        predictions = model.predict(input_data, verbose=0)
        
        # Lay chi so co xac suat cao nhat
        max_index = np.argmax(predictions[0])
        confidence = predictions[0][max_index]
        label_text = LABELS[max_index] if max_index < len(LABELS) else "Unknown"

        # 5. Hien thi ket qua len man hinh (Visualisation)
        # Ve hop chua text
        cv2.rectangle(frame, (10, 10), (300, 60), (0, 0, 0), -1) # Nen den
        
        # Mau sac chu dua tren do tin cay (Xanh la: tin cay cao, Do: tin cay thap)
        color = (0, 255, 0) if confidence > 0.7 else (0, 0, 255)
        
        text = f"{label_text}: {confidence:.2f}"
        cv2.putText(frame, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Ve thanh muc do (Progress bar)
        bar_width = int(confidence * 280)
        cv2.rectangle(frame, (10, 70), (10 + bar_width, 80), color, -1)

        # Hien thi Webcam
        cv2.imshow('Student Engagement Monitor', frame)

        # Nhan 'q' de thoat
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Don dep
    cap.release()
    cv2.destroyAllWindows()
    print("Da thoat chuong trinh.")

if __name__ == "__main__":
    main()
