import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import threading
from PIL import Image, ImageTk, ImageDraw, ImageFont
import time
from collections import defaultdict
import qrcode
import os
import pandas as pd
from datetime import datetime
import serial
import serial.tools.list_ports
import torch
from torchvision import transforms
import torch
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression
from yolov5.utils.augmentations import letterbox

class YOLOModel:
    def __init__(self, weights_path, device='cpu'):
        self.device = device
        self.model = DetectMultiBackend(weights_path, device=device)
        self.stride = self.model.stride
        
    def detect(self, image, conf_thres=0.7, iou_thres=0.45):
        # Tiền xử lý ảnh
        img = letterbox(image, 640, stride=self.stride)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        
        # Chuẩn hóa ảnh
        img = torch.from_numpy(img).to(self.device)
        img = img.float() / 255.0
        
        # Inference
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
            
        pred = self.model(img)
        pred = non_max_suppression(pred, conf_thres, iou_thres)
        
        # Xử lý kết quả
        detections = []
        for det in pred:
            if len(det):
                for *xyxy, conf, cls in reversed(det):
                    detections.append({
                        'bbox': [int(x) for x in xyxy],
                        'confidence': float(conf),
                        'class_id': int(cls)
                    })
        return detections
def update_frame():
    global current_frame, should_capture, current_detections, processing_confirmation
    
    while is_running:
        # Kiểm tra tín hiệu từ Arduino
        if arduino_connected and arduino_serial.in_waiting > 0:
            data = arduino_serial.readline().decode('utf-8').strip()
            if data == "CONFIRM" and not processing_confirmation:
                processing_confirmation = True
                root.after(0, process_confirmation)
        
        ret, frame = cap.read()
        if not ret:
            break
        
        # Chỉ nhận diện khi có tín hiệu xác nhận
        if should_capture:
            detections = recognize_food_with_yolo(frame)
            
            # Vẽ bounding box và nhãn
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                label = f"{det['food_name']} {det['confidence']:.1f}%"
                
                # Vẽ bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                current_detections.append(det['food_name'])
        
        current_frame = frame
        update_gui()
        
        if cv2.waitKey(1) == 27:
            break

# Khởi tạo model YOLO
try:
    yolo_model = YOLOModel('C:/Users/ACER/Downloads/bai tiep nha/mon an/best.pt', device='cuda' if torch.cuda.is_available() else 'cpu')
    print("YOLO model loaded successfully!")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit()

def recognize_food_with_yolo(image):
    """Kết hợp cả YOLO và model hiện tại để nhận diện chính xác hơn"""
    # Nhận diện bằng YOLO trước
    yolo_detections = yolo_model.detect(image)
    
    final_detections = []
    for det in yolo_detections:
        x1, y1, x2, y2 = det['bbox']
        food_roi = image[y1:y2, x1:x2]
        
        # Sử dụng model hiện tại để phân loại món ăn
        img_input = preprocess_image(food_roi)
        color_input = extract_color_features(food_roi)
        preds = model.predict([img_input, color_input], verbose=0)
        idx = np.argmax(preds)
        confidence = np.max(preds) * 0.7 + det['confidence'] * 0.3  # Kết hợp confidence
        
        food_names = list(FOOD_DATA.keys())
        if 0 <= idx < len(food_names) and confidence > 65:  # Ngưỡng confidence cao hơn
            final_detections.append({
                'food_name': food_names[idx],
                'confidence': confidence,
                'bbox': det['bbox']
            })
    
    return final_detections

# 1. Fix lỗi DepthwiseConv2D
class FixedDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)
        super().__init__(*args, **kwargs)

# 2. Danh sách món ăn với giá cả và dinh dưỡng
FOOD_DATA = {
    "Ca hu kho": {"price": 15000, "calories": 124, "protein": 24},
    "Canh cai": {"price": 5000, "calories": 56, "protein": 4},
    "Canh chua": {"price": 7000, "calories": 112, "protein": 2.2},  
    "Com trang": {"price": 3000, "calories": 130, "protein": 2.6},
    "Dau hu sot": {"price": 10000, "calories": 120, "protein": 7},
    "Ga chien": {"price": 15000, "calories": 246, "protein": 30},
    "Rau muong xao": {"price": 5000, "calories": 12, "protein": 2.7, "fiber": 2.2},
    "Thit kho trung": {"price": 15000, "calories": 315, "protein": 25},
    "Trung chien": {"price": 5000, "calories": 135, "protein": 12},
    "Xuc xich": {"price": 10000, "calories": 325, "protein": 18}
}

# Thông tin thanh toán QR
QR_INFO = {
    "bank": "MB Bank",
    "account": "0349428614",
    "name": "LE TIEN TIEP",
    "amount": "",
    "note": "Thanh toan don hang"
}

# Biến toàn cục
cap = None
is_running = True  # Mặc định bật camera khi khởi động
current_frame = None
detected_foods = defaultdict(int)
current_detections = []
qr_image = None
all_bills_history = []
arduino_serial = None
arduino_connected = False
should_capture = False
processing_confirmation = False

# 3. Load model
try:
    model = load_model(
        'C:/Users/ACER/Downloads/bai tiep nha/mon an/food_model_final.h5',
        custom_objects={'DepthwiseConv2D': FixedDepthwiseConv2D},
        compile=False
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f" {e}")
    exit()

class ColorExtractor:
    """Lớp trích xuất 6 màu chính (18 features)"""
    def __init__(self, n_colors=6):
        self.n_colors = n_colors
    
    def extract(self, image):
        """Trích xuất 6 màu chính dùng K-means"""
        pixels = image.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
        _, _, colors = cv2.kmeans(
            pixels, self.n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        colors = np.array(colors)
        colors = sorted(colors, key=lambda c: sum(c), reverse=True)
        return np.ravel(colors)

color_extractor = ColorExtractor()

def preprocess_image(image):
    """Tiền xử lý ảnh đầu vào cho model"""
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img_to_array(img)
    img = preprocess_input(img)
    return np.expand_dims(img, axis=0)

def extract_color_features(image):
    """Trích xuất đặc trưng màu sắc (18 features)"""
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(rgb_image, (224, 224))
    colors = color_extractor.extract(resized_image)
    return np.expand_dims(colors, axis=0)

def recognize_food(image):
    """Nhận diện món ăn với 2 đầu vào (ảnh + màu)"""
    try:
        img_input = preprocess_image(image)
        color_input = extract_color_features(image)
        preds = model.predict([img_input, color_input], verbose=0)
        idx = np.argmax(preds)
        confidence = np.max(preds) * 100
        
        food_names = list(FOOD_DATA.keys())
        if 0 <= idx < len(food_names):
            return food_names[idx], confidence
        return None, 0
        
    except Exception as e:
        print(f"{e}")
        return None, 0

def connect_arduino():
    """Tự động kết nối với Arduino nếu có"""
    global arduino_serial, arduino_connected
    
    # Tìm cổng COM của Arduino
    ports = serial.tools.list_ports.comports()
    arduino_port = None
    for port in ports:
        if 'Arduino' in port.description or 'USB Serial Device' in port.description:
            arduino_port = port.device
            break
    
    if arduino_port is None:
        print("Không tìm thấy Arduino, chỉ sử dụng nút xác nhận trên giao diện")
        return False
    
    try:
        arduino_serial = serial.Serial(arduino_port, 9600, timeout=1)
        time.sleep(2)  # Chờ kết nối ổn định
        arduino_connected = True
        print(f"Đã kết nối với Arduino tại {arduino_port}")
        return True
    except Exception as e:
        print(f"Không thể kết nối Arduino: {e}, chỉ sử dụng nút xác nhận trên giao diện")
        return False

def start_camera():
    """Khởi động camera ngay khi chương trình bắt đầu"""
    global cap, current_frame
    
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        messagebox.showerror("Lỗi", "Không thể mở camera!")
        return
    
    # Thử kết nối Arduino
    connect_arduino()
    
    def update_frame():
        global current_frame, should_capture, current_detections, processing_confirmation
        
        while is_running:
            # Kiểm tra tín hiệu từ Arduino nếu có kết nối
            if arduino_connected and arduino_serial.in_waiting > 0:
                data = arduino_serial.readline().decode('utf-8').strip()
                if data == "CONFIRM" and not processing_confirmation:
                    processing_confirmation = True
                    root.after(0, process_confirmation)
            
            ret, frame = cap.read()
            if not ret:
                break
            
            # Chỉ nhận diện khi có tín hiệu xác nhận
            if should_capture:
                food_name, confidence = recognize_food(frame)
                if food_name and confidence > 70:
                    h, w = frame.shape[:2]
                    box_x1, box_y1 = 50, 50
                    box_x2, box_y2 = w-50, h-50
                    
                    current_detections.append(food_name)
                    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 0), 2)
                    cv2.putText(frame, food_name, (box_x1, box_y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            current_frame = frame
            update_gui()
            
            if cv2.waitKey(1) == 27:
                break
    
    camera_thread = threading.Thread(target=update_frame, daemon=True)
    camera_thread.start()

def update_gui():
    """Cập nhật hình ảnh camera lên giao diện"""
    if current_frame is None:
        return
    
    img = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = img.resize((400, 300))
    img = ImageTk.PhotoImage(image=img)
    
    camera_panel.config(image=img)
    camera_panel.image = img

def update_detected_list():
    """Cập nhật danh sách món ăn đã nhận diện"""
    detected_list.delete(0, tk.END)
    from collections import Counter
    food_counts = Counter(current_detections)
    for food, count in food_counts.items():
        price = FOOD_DATA[food]["price"]
        detected_list.insert(tk.END, f"{food} x{count}: {price:,}VND")

def process_confirmation():
    """Xử lý khi có xác nhận từ nút vật lý hoặc phần mềm"""
    global should_capture, processing_confirmation, current_detections
    
    should_capture = True
    current_detections = []  # Reset danh sách nhận diện
    
    # Chờ 1 giây để camera có đủ thời gian nhận diện
    time.sleep(1)
    
    should_capture = False
    processing_confirmation = False
    
    # Cập nhật danh sách và tạo hóa đơn
    update_detected_list()
    generate_bill()

def generate_bill():
    """Tạo hóa đơn thanh toán với các món đã nhận diện"""
    global detected_foods
    
    if not current_detections:
        messagebox.showwarning("Cảnh báo", "Chưa có món ăn nào được nhận diện!")
        return
    
    # Cập nhật số lượng món ăn đã nhận diện
    detected_foods.clear()
    for food in current_detections:
        detected_foods[food] += 1
    
    update_bill()

def update_bill():
    """Cập nhật hóa đơn thanh toán và QR code"""
    global qr_image, all_bills_history
    
    total = 0
    total_calories = 0
    total_protein = 0
    bill_items = []
    
    for food, count in detected_foods.items():
        food_info = FOOD_DATA[food]
        price = food_info["price"]
        subtotal = price * count
        calories = food_info.get("calories", 0) * count
        protein = food_info.get("protein", 0) * count
        
        bill_items.append({
            "Món ăn": food,
            "Số lượng": count,
            "Đơn giá": f"{price:,}VND",
            "Thành tiền": f"{subtotal:,}VND",
            "Calories": f"{calories}kcal",
            "Protein": f"{protein}g"
        })
        total += subtotal
        total_calories += calories
        total_protein += protein
    
    if bill_items:
        bill_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        all_bills_history.append({
            "Thời gian": bill_time,
            "Chi tiết": bill_items.copy(),
            "Tổng cộng": f"{total:,}VND",
            "Tổng Calories": f"{total_calories}kcal",
            "Tổng Protein": f"{total_protein}g"
        })
    
    bill_text = " HÓA ĐƠN \n\n"
    for item in bill_items:
        bill_text += (f"{item['Món ăn']} x{item['Số lượng']}: {item['Thành tiền']}\n"
                     f"   - Calories: {item['Calories']}\n"
                     f"   - Protein: {item['Protein']}\n")
    
    bill_text += (f"\nTổng số tiền cần thanh toán : {total:,}VND\n"
                 f"Tổng Calories: {total_calories}kcal\n"
                 f"Tổng Protein: {total_protein}g")
    
    bill_label.config(text=bill_text)
    
    if total > 0:
        generate_qr_code(total)
        qr_label.config(image=qr_image)
        qr_label.image = qr_image
        payment_info = f"Chuyển khoản đến:\n{QR_INFO['name']}\n{QR_INFO['bank']}\nSTK: {QR_INFO['account']}"
        payment_label.config(text=payment_info)

def generate_qr_code(total_amount):
    """Tạo mã QR chứa thông tin thanh toán"""
    global qr_image
    
    qr_content = f"bank://{QR_INFO['bank']}?account={QR_INFO['account']}&name={QR_INFO['name']}&amount={total_amount}&note={QR_INFO['note']}"
    
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(qr_content)
    qr.make(fit=True)
    
    qr_img = qr.make_image(fill_color="black", back_color="white")
    qr_img = qr_img.resize((200, 200))
    
    draw = ImageDraw.Draw(qr_img)
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()
    
    draw.text((10, 180), f"STK: {QR_INFO['account']}", font=font, fill="black")
    draw.text((10, 195), f"Ngân hàng: {QR_INFO['bank']}", font=font, fill="black")
    
    qr_image = ImageTk.PhotoImage(qr_img)
    return qr_img

def clear_all():
    """Xóa tất cả thông tin"""
    global detected_foods, current_detections, qr_image, should_capture
    detected_foods.clear()
    current_detections = []
    qr_image = None
    should_capture = False
    update_detected_list()
    bill_label.config(text=" HÓA ĐƠN \n\nChưa có thông tin")
    qr_label.config(image=None)
    payment_label.config(text="")

def on_closing():
    """Xử lý khi đóng cửa sổ"""
    global is_running, cap, arduino_serial
    is_running = False
    if cap is not None:
        cap.release()
    if arduino_serial is not None:
        arduino_serial.close()
    
    export_to_excel()
    root.destroy()

def export_to_excel():
    """Xuất dữ liệu ra file Excel"""
    if not all_bills_history:
        print("Không có dữ liệu để xuất Excel")
        return
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data = []
    for bill in all_bills_history:
        for item in bill["Chi tiết"]:
            data.append({
                "Món ăn": item["Món ăn"],
                "Số lượng": item["Số lượng"],
                "Đơn giá": item["Đơn giá"],
                "Thành tiền": item["Thành tiền"],
            })
    
    df = pd.DataFrame(data)
    totals = {
        "Món ăn": "TỔNG CỘNG",
        "Số lượng": "",
        "Đơn giá": "",
        "Thành tiền": all_bills_history[-1]["Tổng cộng"],
    }
    df = pd.concat([df, pd.DataFrame([totals])], ignore_index=True)
    now = datetime.now()
    filename = os.path.join(current_dir, f"hoa_don_{now.day}_{now.month}_{now.year}_{now.hour}h{now.minute}.xlsx")
    
    try:
        df.to_excel(filename, index=False)
        print(f"Đã xuất file Excel thành công: {filename}")
    except Exception as e:
        print(f"Lỗi khi xuất file Excel: {e}")

# Tạo giao diện
root = tk.Tk()
root.title("Hệ Thống Nhận Diện Món Ăn Tích Hợp Arduino")
root.geometry("1200x800")

# Frame chứa camera và danh sách món ăn
left_frame = tk.Frame(root, width=500, height=700)
left_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH)

# Panel hiển thị camera
camera_panel = tk.Label(left_frame)
camera_panel.pack(pady=5)

# Danh sách món ăn đã nhận diện
detected_list = tk.Listbox(left_frame, height=10, font=("Arial", 12))
detected_list.pack(fill=tk.BOTH, expand=True, pady=5)

# Frame chứa nút điều khiển (chỉ còn 2 nút)
button_frame = tk.Frame(left_frame)
button_frame.pack(pady=10)

clear_btn = tk.Button(button_frame, text="Xóa Tất Cả",
                     command=clear_all,
                     font=("Arial", 12),
                     bg="red", fg="white",
                     width=15)
clear_btn.pack(side=tk.LEFT, padx=5)

confirm_btn = tk.Button(button_frame, text="Xác Nhận",
                       command=process_confirmation,
                       font=("Arial", 12),
                       bg="blue", fg="white",
                       width=15)
confirm_btn.pack(side=tk.LEFT, padx=5)

# Frame hiển thị hóa đơn và QR
right_frame = tk.Frame(root, width=700, height=700, bg="white", bd=2, relief=tk.SUNKEN)
right_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH, expand=True)

# Frame hóa đơn
bill_frame = tk.Frame(right_frame, bg="white")
bill_frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

bill_label = tk.Label(bill_frame, text=" HÓA ĐƠN \n\nChưa có thông tin", 
                     font=("Arial", 14), justify=tk.LEFT, bg="white")
bill_label.pack(anchor=tk.NW)

# Frame QR code
qr_frame = tk.Frame(right_frame, bg="white")
qr_frame.pack(pady=20)

qr_label = tk.Label(qr_frame, bg="white")
qr_label.pack()

payment_label = tk.Label(qr_frame, text="", font=("Arial", 12), bg="white")
payment_label.pack(pady=10)

# Khởi động camera ngay khi chương trình bắt đầu
start_camera()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
