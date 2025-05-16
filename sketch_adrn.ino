const int buttonPin = 2;  // Nút nhấn kết nối với chân 2
int buttonState = 0;      // Trạng thái hiện tại của nút
int lastButtonState = 0;  // Trạng thái trước đó của nút

void setup() {
  Serial.begin(9600);
  pinMode(buttonPin, INPUT_PULLUP);  // Sử dụng điện trở kéo lên nội
}

void loop() {
  // Đọc trạng thái nút nhấn
  buttonState = digitalRead(buttonPin);
  
  // Kiểm tra nếu nút được nhấn (trạng thái LOW do PULLUP)
  if (buttonState != lastButtonState) {
    if (buttonState == LOW) {
      // Gửi tín hiệu xác nhận qua Serial
      Serial.println("CONFIRM");
    }
    delay(50);  // Debounce delay
  }
  
  // Lưu trạng thái hiện tại để so sánh lần sau
  lastButtonState = buttonState;
}