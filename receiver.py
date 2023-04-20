import  RPi.GPIO as GPIO
from time import sleep
GPIO.setmode(GPIO.BOARD)

pin = 11
GPIO.setup(pin,GPIO.IN)
lives = 3
while lives != 0:
    if GPIO.input(pin) == 1:
        print("shot")
        lives -=1
        sleep(3)

    print("safe")
print("It fucking works")
