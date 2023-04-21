from dronekit import connect
import time

# Connect to the Vehicle (in this case serial port)
vehicle = connect('/dev/serial0', baud=57600,  wait_ready=False)

# Letting the vehicle boot up
time.sleep(5)

# Get some vehicle attributes (state)
print ("Get some vehicle attribute values:")
print (" GPS:", vehicle.gps_0)
print (" Battery:", vehicle.battery)
print (" Last Heartbeat:", vehicle.last_heartbeat)
print (" Is Armable?:", vehicle.is_armable)
print (" System status:", vehicle.system_status.state)
print (" Mode:", vehicle.mode.name)    # settable

# Close vehicle object before exiting script
vehicle.close()


print("Completed")
