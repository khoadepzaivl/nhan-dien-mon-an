## Các Thư Viện Cần Cài Đặt

Để chạy đoạn code này, bạn cần cài đặt các thư viện Python sau. Bạn có thể sử dụng `pip` để cài đặt hầu hết các thư viện này.

### Thư Viện Chính

-   **opencv-python:** Xử lý ảnh và video.
    ```bash
    pip install opencv-python
    ```
-   **numpy:** Tính toán số học và mảng đa chiều.
    ```bash
    pip install numpy
    ```
-   **Pillow (PIL):** Thư viện xử lý ảnh.
    ```bash
    pip install Pillow
    ```
-   **tkinter:** Tạo giao diện người dùng đồ họa (GUI).
    ```bash
    # Trên Debian/Ubuntu
    sudo apt-get install python3-tk
    # Trên CentOS/Fedora
    sudo yum install python3-tkinter
    # Hoặc
    pip install tk
    ```
-   **tensorflow:** Nền tảng học máy đầu cuối mã nguồn mở.
    ```bash
    pip install tensorflow
    # Hoặc nếu có GPU NVIDIA hỗ trợ và đã cài đặt CUDA/cuDNN:
    # pip install tensorflow-gpu
    ```
-   **qrcode:** Tạo mã QR.
    ```bash
    pip install qrcode
    ```
-   **pandas:** Phân tích và xử lý dữ liệu.
    ```bash
    pip install pandas
    ```
-   **pyserial:** Giao tiếp nối tiếp (ví dụ: với Arduino).
    ```bash
    pip install pyserial
    ```
-   **torch, torchvision, torchaudio:** Thư viện PyTorch cho deep learning.
    ```bash
    pip install torch torchvision torchaudio
    # Hoặc cài đặt phiên bản phù hợp với CUDA nếu bạn có GPU:
    # pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu](https://download.pytorch.org/whl/cu)<phiên_bản_cuda>
    # (Thay <phiên_bản_cuda> bằng phiên bản CUDA của bạn, ví dụ: cu118)
    ```

### Thư Viện YOLOv5

-   **yolov5:** Để sử dụng YOLOv5, bạn cần **clone repository từ GitHub** và cài đặt các dependencies của nó.
    ```bash
    git clone [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
    cd yolov5
    pip install -r requirements.txt
    ```
    Đảm bảo rằng đường dẫn đến file weights YOLO (`best.pt`) trong code của bạn là chính xác.

### Lưu Ý Quan Trọng
    Sử dụng ctrinh.py để chạy
-   Đảm bảo bạn đang sử dụng **Python 3.10.
-   Đối với các thư viện liên quan đến GPU (`tensorflow-gpu`, `torch` với CUDA), hãy chắc chắn rằng bạn đã cài đặt driver card đồ họa, CUDA và cuDNN tương thích.
-   Nếu gặp vấn đề với `tkinter` trên Linux, hãy sử dụng các lệnh cài đặt hệ thống (`apt-get`, `yum`).
-   Đảm bảo tất cả các model về máy, khi chạy đoạn code hãy thay đúng đường dẫn model tải ở máy vào các dòng load model
-   Khi kết nối adruino và nút ấn bật lí, nhớ kiểm tra dây COM và nối dây theo hướng dẫn của đoạn code
-   Có thể tăng cường dữ liệu và train thêm nếu muốn vì đã có sẵn 1 phần dataset

Sau khi cài đặt đầy đủ các thư viện này, bạn có thể chạy đoạn code được cung cấp.
