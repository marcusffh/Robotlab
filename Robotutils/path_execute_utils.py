# Robotutils/path_exec_utils.py
from math import atan2, hypot, pi
from Robotutils.CalibratedRobot import CalibratedRobot

def _wrap_to_pi(a):
    while a >  pi: a -= 2*pi
    while a < -pi: a += 2*pi
    return a

def execute_path(
    bot,
    path,
    drive_speed=64,       # 0..127, your default is 64
    turn_speed=64,        # 0..127
    heading_tol_deg=5.0,  # don't turn if within this tolerance
    min_seg_len=0.05,     # skip very short segments (m)
    settle_s=0.05         # small pause after each move
):
    """
    Execute waypoints using ONLY CalibratedRobot:
      - bot.turn_angle(angleDeg, speed=...)
      - bot.drive_distance(meters, direction=bot.FORWARD, speed=...)
      - bot.stop()

    Assumptions:
      - Path is [(x0,z0), (x1,z1), ...] in meters.
      - Robot starts at (0,0) with heading = 0 rad (along +x in your local map).
      - Segment heading computed with atan2(dz, dx).
    """
    if not path or len(path) < 2:
        print("execute_path: nothing to do.")
        return

    import time
    heading = pi/2 # internal estimate in radians

    try:
        for (x0, z0), (x1, z1) in zip(path[:-1], path[1:]):
            dx, dz = x1 - x0, z1 - z0
            seg_len = hypot(dx, dz)
            if seg_len < min_seg_len:
                continue

            # 1) Turn to face the segment
            desired = atan2(dz, dx)                 # absolute desired heading
            dtheta = _wrap_to_pi(desired - heading) # turn needed
            dtheta_deg = dtheta * 180.0 / pi
            if abs(dtheta_deg) > heading_tol_deg:
                bot.turn_angle(dtheta_deg, speed=turn_speed)
                heading = _wrap_to_pi(heading + dtheta)
                time.sleep(settle_s)

            # 2) Drive straight the segment length
            bot.drive_distance(seg_len, direction=bot.FORWARD, speed=drive_speed)
            time.sleep(settle_s)

        bot.stop()
        print("execute_path: done.")
    finally:
        bot.stop()
