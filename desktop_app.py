import sys
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
from datetime import datetime
from collections import deque

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QProgressBar, 
                             QGroupBox, QFileDialog, QMessageBox)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer, QDate
from PyQt6.QtGui import QImage, QPixmap, QAction

import pyqtgraph as pg

# ==========================================
# CẤU HÌNH & UTILS
# ==========================================
MODEL_PATH = 'Student Engagement Model.h5'
LABELS = ["Engaged", "Confused", "Bored", "Drowsy", "Frustrated", "Looking Away"]
SMOOTHING_WINDOW = 15  # Số lượng frame để lấy trung bình (giam jitter)

def get_current_time_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ==========================================
# WORKER THREAD: XU LY VIDEO & AI
# ==========================================
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    update_data_signal = pyqtSignal(str, float) # Label, Confidence

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.model = None
        self.prediction_history = deque(maxlen=SMOOTHING_WINDOW)

    def load_model(self):
        if self.model is None:
            print("Loading model in thread...")
            try:
                self.model = tf.keras.models.load_model(MODEL_PATH)
                print("Model loaded!")
            except Exception as e:
                print(f"Error loading model: {e}")
                self.stop()

    def run(self):
        self.load_model()
        cap = cv2.VideoCapture(0)
        
        while self._run_flag:
            ret, frame = cap.read()
            if ret:
                # 1. Prediction Logic
                resized = cv2.resize(frame, (256, 256))
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                norm = rgb.astype('float32') / 255.0
                input_data = np.expand_dims(norm, axis=0)
                
                if self.model:
                    preds = self.model.predict(input_data, verbose=0)[0]
                    self.prediction_history.append(preds)
                    
                    # SMOOTHING: Tinh trung binh cac du doan gan nhat
                    avg_preds = np.mean(self.prediction_history, axis=0)
                    max_idx = np.argmax(avg_preds)
                    
                    label = LABELS[max_idx] if max_idx < len(LABELS) else "Unknown"
                    conf = avg_preds[max_idx]
                    
                    self.update_data_signal.emit(label, float(conf))

                # 2. Update UI Image
                self.change_pixmap_signal.emit(frame)
            else:
                break
                
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

# ==========================================
# MAIN WINDOW
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EduVision Pro - Student Engagement Monitor")
        self.resize(1200, 700)
        self.history_data = [] # Luu du lieu de xuat bao cao
        
        # Setup UI
        self.init_ui()
        
        # Thread setup
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.update_data_signal.connect(self.update_stats)
        self.thread.start()

        # Timer cho bieu do (cap nhat moi giay thay vi moi frame de nhe may)
        self.graph_timer = QTimer()
        self.graph_timer.timeout.connect(self.update_graph)
        self.graph_timer.start(1000) # 1s/lan
        
        self.current_label = "Initializing..."
        self.current_conf = 0.0

    def init_ui(self):
        # Main Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- LEFT COLUMN: CAMERA ---
        left_layout = QVBoxLayout()
        
        # Video Label
        self.image_label = QLabel("Camera Loading...")
        self.image_label.setFixedSize(640, 480)
        self.image_label.setStyleSheet("background-color: #000; color: #FFF; border: 2px solid #444;")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(self.image_label)
        
        # Controls
        control_group = QGroupBox("Controls")
        control_layout = QHBoxLayout()
        
        self.btn_stop = QPushButton("STOP CAMERA")
        self.btn_stop.clicked.connect(self.toggle_camera)
        self.btn_stop.setStyleSheet("background-color: #d9534f; color: white; padding: 10px; font-weight: bold;")
        
        self.btn_export = QPushButton("EXPORT REPORT (CSV)")
        self.btn_export.clicked.connect(self.export_report)
        self.btn_export.setStyleSheet("background-color: #0275d8; color: white; padding: 10px;")

        control_layout.addWidget(self.btn_stop)
        control_layout.addWidget(self.btn_export)
        control_group.setLayout(control_layout)
        left_layout.addWidget(control_group)
        
        main_layout.addLayout(left_layout)

        # --- RIGHT COLUMN: STATS & GRAPH ---
        right_layout = QVBoxLayout()
        
        # 1. Current Status Box
        stats_group = QGroupBox("Live Status")
        stats_layout = QVBoxLayout()
        
        self.lbl_status = QLabel("WAITING...")
        self.lbl_status.setStyleSheet("font-size: 36px; font-weight: bold; color: #f0ad4e;")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.progress_conf = QProgressBar()
        self.progress_conf.setRange(0, 100)
        self.progress_conf.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
                height: 30px;
            }
            QProgressBar::chunk {
                background-color: #5cb85c; 
            }
        """)
        
        stats_layout.addWidget(self.lbl_status)
        stats_layout.addWidget(QLabel("Confidence Level:"))
        stats_layout.addWidget(self.progress_conf)
        stats_group.setLayout(stats_layout)
        right_layout.addWidget(stats_group)

        # 2. Live Graph
        graph_group = QGroupBox("Engagement History (Last 60s)")
        graph_layout = QVBoxLayout()
        
        self.graph_widget = pg.PlotWidget()
        self.graph_widget.setBackground('w')
        self.graph_widget.setTitle("Engagement Score Over Time", color="k", size="12pt")
        self.graph_widget.setYRange(0, 1)
        self.graph_widget.showGrid(x=True, y=True)
        
        # Chung ta se ve duong 'Confidence' cua trang thai hien tai
        self.time_data = list(range(60))
        self.val_data = [0.0] * 60
        self.data_line = self.graph_widget.plot(self.time_data, self.val_data, pen=pg.mkPen(color='b', width=2))
        
        graph_layout.addWidget(self.graph_widget)
        graph_group.setLayout(graph_layout)
        right_layout.addWidget(graph_group)

        main_layout.addLayout(right_layout)

    def update_image(self, cv_img):
        """Convert CV image to Qt image and display"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        p = convert_to_Qt_format.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
        self.image_label.setPixmap(QPixmap.fromImage(p))

    def update_stats(self, label, conf):
        """Nhan data tu Thread va cap nhat UI"""
        self.current_label = label
        self.current_conf = conf
        
        # Cap nhat mau sac dua tren label
        color_map = {
            "Engaged": "#5cb85c", # Green
            "Bored": "#d9534f",   # Red
            "Drowsy": "#f0ad4e",  # Orange
            "Looking Away": "#777", # Grey
        }
        color = color_map.get(label, "#0275d8") # Blue default
        
        self.lbl_status.setText(label.upper())
        self.lbl_status.setStyleSheet(f"font-size: 36px; font-weight: bold; color: {color};")
        self.progress_conf.setValue(int(conf * 100))
        
        # Luu vao history
        self.history_data.append({
            "Timestamp": get_current_time_str(),
            "Label": label,
            "Confidence": conf
        })

    def update_graph(self):
        """Cap nhat bieu do moi giay"""
        # Day gia tri moi nhat vao list, xoa gia tri cu
        self.val_data = self.val_data[1:]  # Remove first
        self.val_data.append(self.current_conf)  # Add new
        
        self.data_line.setData(self.time_data, self.val_data)

    def toggle_camera(self):
        if self.thread.isRunning():
            self.thread.stop()
            self.btn_stop.setText("START CAMERA")
            self.lbl_status.setText("PAUSED")
        else:
            self.thread = VideoThread()
            self.thread.change_pixmap_signal.connect(self.update_image)
            self.thread.update_data_signal.connect(self.update_stats)
            self.thread.start()
            self.btn_stop.setText("STOP CAMERA")

    def export_report(self):
        if not self.history_data:
            QMessageBox.warning(self, "No Data", "No data recorded yet.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV Files (*.csv)")
        if path:
            df = pd.DataFrame(self.history_data)
            df.to_csv(path, index=False)
            QMessageBox.information(self, "Success", f"Report saved to {path}")

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Dark Theme style
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
