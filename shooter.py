import  RPi.GPIO as GPIO
from time import sleep
GPIO.setmode(GPIO.BCM)

pin = 3
GPIO.setup(pin,GPIO.OUT)
delays = 3
while True:
    if delays == 0:
        break
    inp = input("press enter to shoot")
    GPIO.output(pin,GPIO.HIGH)
    delays -= 1
    print(1)
    GPIO.output(pin,GPIO.LOW)
    sleep(3)
print(0)
GPIO.cleanup()
