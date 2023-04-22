from dronekit import connect, VehicleMode,LocationGlobalRelative,APIException, mavutil
import time
import socket
import math
import threading
from pymavlink import mavutil


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

def move_gps():
	vehicle.mode = VehicleMode("AUTO")
	while vehicle.mode.name != "AUTO":
		print("Waiting for vehicle to enter AUTO mode")
		time.sleep(1)
	current_latitude = vehicle.location.global_relative_frame.lat
	current_longitude = vehicle.location.global_relative_frame.lon
	current_heading = math.radians(vehicle.heading)
		
	distance = 10
	new_lat = current_latitude + (distance * math.cos(current_heading)) / 111111
	new_long = current_longitude + (distance * math.sin(current_heading)) / (111111 * math.cos(current_latitude))

	target_location = LocationGlobalRelative(new_lat,new_long,0)
	
	vehicle.simple_goto(target_location)
	
	while True:
		current_location = vehicle.location.global_relative_frame
		distance_to_target = target_location.distance_to(current_location)
		if distance_to_target < 1:
			print("Reached the target location")
			break
		time.sleep(1)
	
####main####	
vehicle = connectMyCopter()

vehicle.wait_ready('autopilot_version')
print('Autopilot version:', vehicle.version)

arm()

print("Start Run")
move_gps()
vehicle.close()
