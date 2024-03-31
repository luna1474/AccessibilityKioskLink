import time
import usb_hid
from adafruit_hid.mouse import Mouse
import board
import busio
import digitalio

# 마우스 객체 생성
mouse = Mouse(usb_hid.devices)

# UART 초기화
uart = busio.UART(board.GP0, board.GP1, baudrate=9600)  # UART 설정에 따라 핀을 조정해야 합니다.

# 클릭 함수 정의
def click(x, y):
    mouse.move(x=-5000, y=-5000) 
    mouse.move(x=x, y=y)
    mouse.click(Mouse.LEFT_BUTTON)

# Main loop
while True:
    data = uart.readline() # UART로부터 명령을 읽어옴
    if data is not None:
        print(data)
        command = data.strip().decode()
        print(command)
        if command.startswith("click"):  # "click"으로 시작하는 명령이면
            parts = command.split(",")    # 쉼표를 기준으로 명령을 분할
            x = int(parts[1])
            y = int(parts[2])
            click(x, y)  # 클릭 함수 호출
    
    # Add a small delay to avoid flooding the UART input
    time.sleep(0.1)
