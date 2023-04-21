from dronekit import connect, VehicleMode,LocationGlobalRelative,APIException
import time
import socket
import math
import argparse


#########FUNCTIONS#################

#connection_string='/dev/serial0'
#baud=57600

def connectMyCopter():

	# Connect to the Vehicle (in this case serial port)
	vehicle = connect('/dev/serial0', baud=57600,  wait_ready=False)
	return vehicle


def arm():
	# while vehicle.is_armable!=True:
		# print("Waiting for vehicle to become armable.")
		# time.sleep(1)
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

def run():
	vehicle.groundspeed = 2
	location = LocationGlobalRelative(10, 0, 0)
	vehicle.simple_goto(location)
	return None

def stop():
	vehicle.groundspeed = 0
	return None



##########MAIN EXECUTABLE###########


vehicle = connectMyCopter()

vehicle.wait_ready('autopilot_version')
print('Autopilot version:', vehicle.version)

# Arming the vehicle and setting it in MANUAL mode
arm()
time.sleep(1)

# Running the vehicle for 4 seconds
print("Start run!")
run()
time.sleep(4)

# Stopping the vehicle
# print("Stop!")
# stop()

# Close vehicle object before exiting script
vehicle.close()
