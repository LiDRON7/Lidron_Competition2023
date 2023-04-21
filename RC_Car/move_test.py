import os
import time

# List of commands
start_command = "sudo mavproxy.py --master=/dev/serial0 --baudrate 57600 --aircraft MyCopter"

manual_mode = "mode MANUAL"

arm_throttle = "arm throttle"

throttle_foward = "rc 3 1500"  # Going foward at half speed

throttle_stop = "rc 3 0"

# Create the connection
print("Making the connection . . .")
os.system(start_command)

# Wait a few seconds to let the system boot up
time.sleep(5)
print("Detected heartbeat!")
print("\n--COMMENCE OPERATION--")

# Change to manual mode
os.system(manual_mode)
time.sleep(1)

# Arm throttle
os.system(arm_throttle)
time.sleep(1)

# Go foward for 4 seconds
print("Start running!")
os.system(throttle_foward)
time.sleep(4)

# Stop car
print("Stop!")
os.system(throttle_stop)
time.sleep(1)
print("End of script")
