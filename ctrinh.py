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

# 1. Fix lỗi DepthwiseConv2D
class FixedDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)
        super().__init__(*args, **kwargs)

# 2. Danh sách món ăn với giá cả và dinh dưỡng
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
is_running = False
current_frame = None
detected_foods = defaultdict(int)
current_detections = []
reset_flag = False
last_reset_time = 0
qr_image = None
all_bills_history = []

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
detected_boxes = []
def start_camera():
    """Bắt đầu quá trình nhận diện từ camera"""
    global cap, is_running, current_frame, current_detections, reset_flag, last_reset_time
    
    if is_running:
        return
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Lỗi", "Không thể mở camera!")
        return
    
    is_running = True
    
    def update_frame():
        global current_frame, current_detections, reset_flag, last_reset_time, detected_boxes
        
        while is_running:
            ret, frame = cap.read()
            if not ret:
                break
            
            if reset_flag:
                if time.time() - last_reset_time > 10:
                    reset_flag = False
                    current_detections = []
                    detected_boxes = []
                else:
                    remaining = 10 - int(time.time() - last_reset_time)
                    cv2.putText(frame, f"Dang reset... {remaining}s", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    current_frame = frame
                    update_gui()
                    continue
            
            food_name, confidence = recognize_food(frame)
            if food_name and confidence > 70:
                # Xác định vị trí khung (giả lập - thực tế cần detect từ model)
                h, w = frame.shape[:2]
                box_x1, box_y1 = 50, 50
                box_x2, box_y2 = w-50, h-50
                
                # Kiểm tra xem có khung trùng lặp không
                is_new_box = True
                for box in detected_boxes:
                    # Kiểm tra overlap giữa các khung
                    if abs(box[1] - box_x1) < 50 and abs(box[2] - box_y1) < 50:
                        is_new_box = False
                        break
                
                if is_new_box:
                    detected_boxes.append((food_name, box_x1, box_y1, box_x2, box_y2))
                    current_detections.append(food_name)
                    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 0), 2)
                    cv2.putText(frame, food_name, (box_x1, box_y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            current_frame = frame
            update_gui()
            update_detected_list()
            
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
                "Calories": item["Calories"],
                "Protein": item["Protein"]
            })
    
    df = pd.DataFrame(data)
    
    totals = {
        "Món ăn": "TỔNG CỘNG",
        "Số lượng": "",
        "Đơn giá": "",
        "Thành tiền": all_bills_history[-1]["Tổng cộng"],
        "Calories": all_bills_history[-1]["Tổng Calories"],
        "Protein": all_bills_history[-1]["Tổng Protein"]
    }
    df = pd.concat([df, pd.DataFrame([totals])], ignore_index=True)
    
    now = datetime.now()
    filename =os.path.join(current_dir, f"ket ca_{now.hour}h_{now.minute}_{now.day}_{now.month}_{now.year}.xlsx")
    
    try:
        df.to_excel(filename, index=False)
        print(f"Đã xuất file Excel thành công: {filename}")
    except Exception as e:
        print(f"Lỗi khi xuất file Excel: {e}")

def reset_detection():
    """Reset nhận diện cho phần cơm mới"""
    global reset_flag, last_reset_time, current_detections, detected_foods
    reset_flag = True
    last_reset_time = time.time()
    current_detections = []
    detected_foods.clear()
    update_detected_list()
    bill_label.config(text=" HÓA ĐƠN \n\nChưa có thông tin")
    qr_label.config(image=None)
    payment_label.config(text="")
    messagebox.showinfo("Thông báo", "Đang reset cho phần cơm mới. Vui lòng đợi 10 giây.")

def generate_bill():
    """Tạo hóa đơn thanh toán"""
    global detected_foods
    
    if not current_detections:
        messagebox.showwarning("Cảnh báo", "Chưa có món ăn nào được nhận diện!")
        return
    
    # Tự động đếm số lượng theo vị trí khung
    from collections import Counter
    detected_foods = Counter(current_detections)
    update_bill()
    # Cho phép nhập số lượng thủ công cho các món trùng
    popup = tk.Toplevel()
    popup.title("Nhập số lượng")
    
    tk.Label(popup, text="Nhập số lượng cho từng món:").pack()
    
    quantity_entries = {}
    for food in set(current_detections):  # Sử dụng set để lấy món duy nhất
        frame = tk.Frame(popup)
        frame.pack(pady=5)
        tk.Label(frame, text=f"{food}:").pack(side=tk.LEFT)
        quantity_entries[food] = tk.Entry(frame)
        quantity_entries[food].pack(side=tk.LEFT)
        quantity_entries[food].insert(0, str(current_detections.count(food)))  # Đếm số lần xuất hiện
    
    def confirm_quantities():
        detected_foods.clear()
        for food, entry in quantity_entries.items():
            try:
                quantity = int(entry.get())
                detected_foods[food] = quantity
            except ValueError:
                detected_foods[food] = 1
        popup.destroy()
        update_bill()
    
    tk.Button(popup, text="Xác nhận", command=confirm_quantities).pack(pady=10)
def clear_all():
    """Xóa tất cả thông tin"""
    global detected_foods, current_detections, qr_image
    detected_foods.clear()
    current_detections = []
    qr_image = None
    update_detected_list()
    bill_label.config(text=" HÓA ĐƠN \n\nChưa có thông tin")
    qr_label.config(image=None)
    payment_label.config(text="")

def start_recognition():
    """Bắt đầu nhận diện trong thread riêng"""
    global is_running
    if not is_running:
        start_camera()
        start_btn.config(state=tk.DISABLED)
    else:
        messagebox.showinfo("Thông báo", "Hệ thống đang chạy")

def on_closing():
    """Xử lý khi đóng cửa sổ"""
    global is_running, cap
    is_running = False
    if cap is not None:
        cap.release()
    
    export_to_excel()
    root.destroy()

# Tạo giao diện
root = tk.Tk()
root.title("Hệ Thống Nhận Diện Món Ăn với Dinh Dưỡng")
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

# Frame chứa nút điều khiển
button_frame = tk.Frame(left_frame)
button_frame.pack(pady=10)

start_btn = tk.Button(button_frame, text="Bắt Đầu", 
                     command=start_recognition,
                     font=("Arial", 12), 
                     bg="green", fg="white",
                     width=15)
start_btn.pack(side=tk.LEFT, padx=5)

reset_btn = tk.Button(button_frame, text="Thay Đổi Phần Cơm",
                    command=reset_detection,
                    font=("Arial", 12),
                    bg="orange", fg="white",
                    width=15)
reset_btn.pack(side=tk.LEFT, padx=5)

bill_btn = tk.Button(button_frame, text="Xuất Hóa Đơn",
                    command=generate_bill,
                    font=("Arial", 12),
                    bg="blue", fg="white",
                    width=15)
bill_btn.pack(side=tk.LEFT, padx=5)

clear_btn = tk.Button(button_frame, text="Xóa Tất Cả",
                     command=clear_all,
                     font=("Arial", 12),
                     bg="red", fg="white",
                     width=15)
clear_btn.pack(side=tk.LEFT, padx=5)

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

# Xử lý sự kiện đóng cửa sổ
root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()