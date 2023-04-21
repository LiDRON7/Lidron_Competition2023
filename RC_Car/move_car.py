import time
from pimavlink import mavutil

master = mavutil.mavlink_connection("/dev/serial0", baud=57600)

master.arducopter_arm()
master.set_mode_auto()

target_lat = 
target_lon = 
target_alt = 

master.mav_mission_item_send(
    0,
    0,
    0,
    0,
    0,
    0,
    target_lat,
    target_lon,
    target_alt
)

while True:
    msg = master.rev_match(type='MISSION_CURRENT',blockings=True)
    if msg.seq = 1:
        print("target waypoint reach")
        break
    time.sleep(1)
