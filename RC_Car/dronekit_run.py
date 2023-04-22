from dronekit import connect, VehicleMode,LocationGlobalRelative,APIException, mavutil
import time
import socket
import math
import threading
from pymavlink import mavutil


#########FUNCTIONS#################

#connection_string='/dev/serial0'
#baud=57600

def connectMyCopter():

	# Connect to the Vehicle (in this case serial port)
	vehicle = connect('/dev/serial0', baud=57600,  wait_ready=False)
	return vehicle


def arm():
	while vehicle.is_armable!=True:
		print("Waiting for vehicle to become armable.")
		time.sleep(1)
	print("Vehicle is now armable")

	vehicle.mode = VehicleMode("MANUAL")

	while vehicle.mode!='MANUAL':
		print("Waiting for drone to enter MANUAL flight mode")
		time.sleep(1)
	print("Vehicle now in MANUAL MODE!")

	vehicle.armed = True
	time.sleep(1)
	print("Vehicle armed:", vehicle.armed)

	while vehicle.armed==False:
		print("Waiting for vehicle to become armed.")
		time.sleep(1)
	print("Vehicle is now armed.")

	return None



def run_test():
	
	# Set the ground velocity
	vehicle.groundspeed = 10
	
	# Define a distance of 10 meters in front of the vehicle
	distance = 10

	# Define a location that is `distance` meters in front of the vehicle
	location = LocationGlobalRelative(distance, 0, 0)
	
	# Send the vehicle to the new location
	vehicle.simple_goto(location)
	
def run():
	vehicle.channels.overrides['3'] = 1500
	return None

"""def run_until_shot():
	vehicle.channels.overrides['3'] = 1500
	time.sleep(5)
	
	start_time = time.time()
	print(start_time)
	duration = 10
	while :
		elapsed_time = time.time() - start_time
		print(elapsed_time)
		if elapsed_time >= duration:
			print("didnt get shot")
			break
	GPIO.cleanup()
	vehicle.channels.overrides['3'] = 0"""

def stop():
	vehicle.groundspeed = 0
	return None


def move_rover_forward(speed, duration):
    """
    Move the rover forward at a constant velocity for the specified duration.
    """
    velocity_x = speed
    velocity_y = 0
    velocity_z = 0
    
    # Call the send_ned_velocity function
    send_ned_velocity(velocity_x, velocity_y, velocity_z, duration)
    
def move_rover_forward2(speed, duration):
    """
    Move the rover forward at a constant velocity for the specified duration.
    """
    
    # Set forward movement
    forward = 0.5
    
    # Send manual control commands to move the rover forward
    for x in range(0,duration*10):
        vehicle.channels.overrides = {'1': 1500, '2': 1500, '3': 1500, '4': int(1500+forward*speed)} #this is kinda weird
        time.sleep(0.1)



def test_thread():
	import threading
	def keep_vehicle_running():
		while not stop_flag.is_set():
			vehicle.channels.overrides['3']=500
			print("running")
			time.sleep(0.1)
		vehicle.channels.overrides['3'] = 0
	def monitor_pin_input():
		import RPi.GPIO as GPIO
		GPIO.setmode(GPIO.BOARD)
		pin = 11
		GPIO.setup(pin,GPIO.IN)
		while True:
			value = GPIO.input(pin)
			print(value)
			if value == 1:
				stop_flag.set()
				break
		time.sleep(0.1)
		GPIO.cleanup()
	stop_flag = threading.Event()
	pin_thread = threading.Thread(target=monitor_pin_input)
	pin_thread.start()
	
	vehicle_thread = threading.Thread(target=keep_vehicle_running)
	vehicle_thread.start()
	pin_thread.join()

def test2():
	import RPi.GPIO as GPIO
	GPIO.setmode(GPIO.BOARD)
	pin = 11
	GPIO.setup(pin,GPIO.IN)
	while True:
		vehicle.channels.overrides['3']=1500
		value = GPIO.input(pin)
		print(value)
		if value != 1:
			pass
		

def gps_test():
	while True:
		print("GPS: %s", vehicle.gps_0)

			
##########MAIN EXECUTABLE##########


vehicle = connectMyCopter()

vehicle.wait_ready('autopilot_version')
print('Autopilot version:', vehicle.version)

#Arming the vehicle and setting it in MANUAL mode
arm()

#Running the vehicle for 4 seconds
print("Start run!")
test_thread()
# Stopping the vehicle
print("Stop!")
# stop()

# gps_test()

# Close vehicle object before exiting script
vehicle.close()


