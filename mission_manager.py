#!/usr/bin/env python3
"""Step 5: companion-side guard + offboard mission manager (GNSS-denied UAV).

Runs the deployment mission logic on the companion computer over uXRCE-DDS
(the link validated in step 4c). For the step-7 rehearsal it replaces
`sitl_ev_driver.py` — never run both: they would both publish
/fmu/in/vehicle_visual_odometry.

Subscriptions
  /mcl/odom    (nav_msgs/Odometry) — MCL GPS-substitute: pose in the ENU map
               frame (x east / y north of the MAP CENTER) + VIO velocity in
               base_link FLU. Honesty-gated by mcl_node: pose covariance
               >= POSE_GATED_COV means "gated" (MCL not tracking), the twist
               is always honest.
  /mcl/health  (std_msgs/String, 1 Hz JSON) — MCL state (step 3), the guard's
               primary input. Missing topic -> the guard falls back to EV
               freshness only (with a warning), so a rehearsal can run on a
               recorded /mcl/odom replay without the health channel.
  /fmu/out/vehicle_status_v1, /fmu/out/vehicle_local_position_v1,
  /fmu/out/vehicle_land_detected — PX4 state.

Publications
  /fmu/in/vehicle_visual_odometry — EV odometry for EKF2 (ENU/FLU -> NED/FRD,
     EKF2_EV_CTRL=5: horizontal position + 3D velocity; baro owns z). Gated
     poses are encoded as NaN position: zeros + a huge covariance is NOT
     "no data" — EKF2 fuses it and collapses the estimate toward the EV
     origin between honest fixes (~140 xy resets observed, see COMMANDS.md §5).
  /fmu/in/offboard_control_mode + /fmu/in/trajectory_setpoint — OFFBOARD
     velocity/yaw setpoints at 50 Hz.
  /fmu/in/vehicle_command — ARM/DISARM, DO_SET_MODE (OFFBOARD / AUTO.LAND),
     SET_GPS_GLOBAL_ORIGIN (map center; makes vehicle_global_position
     comparable to map lat/lon).
  /mission/state (std_msgs/String, 1 Hz JSON) — guard state + reasons, for the
     rehearsal recorder / operator.

Guard + mission phases are implemented in mission_guard.py (pure logic,
unit-tested by test_mission_guard.py). This file is only the ROS/PX4 glue and
setpoint generation.

Mission modes (param `mission_mode`):
  feedforward — velocity = EV twist rotated by the EV heading: the airframe
     shadows the replayed flight (rehearsal). Altitude is HELD at cruise_alt
     instead of following the bag's body-twist z (that z carried real-flight
     tilt coupling -> sim-only over-climb / late attitude failure, see
     COMMANDS.md §5). EV-stream end (>= STREAM_END_S silence) -> mission done
     -> LAND.
  waypoints — velocity = P-controller toward the next waypoint (param
     `waypoints`, "x,y; x,y; ..." in map ENU metres), yaw along the route,
     altitude held; last waypoint -> mission done -> LAND.

EV frame (param `ev_frame`):
  map       — EV poses are already in the PX4 local NED frame; the local
              frame origin expressed in map ENU is (local_origin_map_x/y)
              (default 0,0 = EKF2 local origin is the map centre). Real
              airframe: set it to the take-off point's map ENU position
              (operator init point minus map centre).
  first_fix — translate EV by the FIRST honest fix (map frame -> sim local
              frame). The SITL rehearsal hack: the Gazebo world origin is
              where the replay's track starts.

Run (SITL rehearsal):
  source /opt/ros/jazzy/setup.bash
  source ~/ros2_ws/install/setup.bash   # px4_msgs (v1.17, prebuilt, Jazzy)
  python3 mission_manager.py --ros-args -p ev_frame:=first_fix

Run (real airframe, waypoint mission, 1 km map-edge margin enforced):
  python3 mission_manager.py --ros-args \
    -p mission_mode:=waypoints \
    -p 'waypoints:=137.0,-185.0; 500.0,-185.0; 500.0,300.0' \
    -p cruise_alt:=50.0 \
    -p local_origin_map_x:=137.0 -p local_origin_map_y:=-185.0
"""

import json
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from nav_msgs.msg import Odometry
from px4_msgs.msg import (OffboardControlMode, TrajectorySetpoint,
                          VehicleCommand, VehicleLandDetected,
                          VehicleLocalPosition, VehicleOdometry,
                          VehicleStatus)
from std_msgs.msg import String

from mission_guard import (DONE, HOLD, LAND, MISSION, RETURN, STREAM_END_S,
                           STATE_NAMES, TAKEOFF, WAIT, WP_RADIUS_M, Guard)

# --------------------------------------------------------------- constants
SQ2 = math.sqrt(0.5)
Q_ENU2NED = (0.0, SQ2, SQ2, 0.0)   # ENU -> NED (180 deg about (1,1,0)/sqrt2)
Q_FLU2FRD = (0.0, 1.0, 0.0, 0.0)   # FLU -> FRD (180 deg about x)

POSE_GATED_COV = 1000.0            # pose covariance >= -> MCL pose is gated
CLIMB_RATE = 2.5                   # takeoff climb rate [m/s]
ALT_KP = 0.5                       # altitude-hold P gain [1/s]
ALT_VZ_MAX = 1.5                   # altitude-hold climb rate clamp [m/s]
SETPOINT_HZ = 50.0
YAW = float('nan')

CMD_ARM_DISARM = 400                       # VEHICLE_CMD_COMPONENT_ARM_DISARM
CMD_DO_SET_MODE = 176                      # VEHICLE_CMD_DO_SET_MODE
CMD_SET_GPS_GLOBAL_ORIGIN = 100000         # PX4-internal: p5/p6 = lat/lon deg
# DO_SET_MODE params: p1 = MAV_MODE_FLAG_CUSTOM_ENABLED(1); p2/p3 = PX4 custom
# main/sub mode. SET_NAV_STATE (100001) acks but never applies (step 4c).
MODE_OFFBOARD = (1.0, 6.0, 0.0)            # main=OFFBOARD
MODE_AUTO_LAND = (1.0, 4.0, 6.0)           # main=AUTO, sub=LAND
NAV_STATE_OFFBOARD = 14                    # VehicleStatus.NAVIGATION_STATE_*
NAV_STATE_AUTO_LAND = 18
ARMING_STATE_ARMED = 2


def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw)


def parse_waypoints(text):
    """'x,y; x,y' (map ENU metres) -> [(x, y), ...]."""
    wps = []
    for chunk in str(text).split(';'):
        chunk = chunk.strip()
        if not chunk:
            continue
        x, y = chunk.split(',')
        wps.append((float(x), float(y)))
    return wps


class MissionManager(Node):

    def __init__(self):
        super().__init__('mission_manager')

        def param(name, default):
            self.declare_parameter(name, default)
            return self.get_parameter(name).value

        # --- mission ----------------------------------------------------
        self.mission_mode = str(param('mission_mode', 'feedforward'))
        self.waypoints = parse_waypoints(param('waypoints', ''))
        if self.mission_mode == 'waypoints' and not self.waypoints:
            raise SystemExit('mission_mode=waypoints needs waypoints:="x,y; ..."')
        # --- guard limits -----------------------------------------------
        self.guard = Guard(
            cruise_alt=param('cruise_alt', 50.0),
            takeoff_max_s=param('takeoff_max_s', 120.0),
            hold_budget_s=param('hold_budget_s', 180.0),
            return_max_s=param('return_max_s', 60.0),
            mission_max_s=param('mission_max_s', 900.0),
            margin_m=param('margin_m', 1000.0),
            map_half_m=param('map_half_m', 2500.0))
        self.v_max = float(param('v_max', 12.0))
        self.preflight_max_s = float(param('preflight_max_s', 180.0))
        # --- EV frame ---------------------------------------------------
        self.ev_frame = str(param('ev_frame', 'map'))
        self.local_origin_map_x = float(param('local_origin_map_x', 0.0))
        self.local_origin_map_y = float(param('local_origin_map_y', 0.0))
        # EKF2 global origin = deployment-map centre (z17_5120.png px 2560,2560)
        self.set_global_origin = bool(param('set_global_origin', True))
        self.map_center_lat = float(param('map_center_lat', 22.8445297))
        self.map_center_lon = float(param('map_center_lon', 114.5242310))
        self.ff_altitude_hold = bool(param('ff_altitude_hold', True))
        # Sim rehearsal only: the bag's body-twist z carries real-flight tilt
        # coupling (~1.6 m/s) which the sim airframe does not have -> the EV
        # vertical velocity pumps EKF2's vz and inflates the baro height
        # innovation (preflight height gate 0.5 -> arming blocked). Real
        # airframe: leave False (the FCU rotates an honest body velocity).
        self.ev_vz_zero = bool(param('ev_vz_zero', False))

        # --- state ------------------------------------------------------
        self.status = None
        self.status_t = 0.0
        self.lpos = None
        self.landed = False
        self.health = None
        self.health_t = 0.0
        self.ev = None                 # last /mcl/odom
        self.ev_t = None               # node clock at its arrival
        self.first_ev_t = None
        self.p_fix = None              # (x, y) map ENU of the first honest fix
        self.ev_map_pos = None         # honest EV map ENU (guard input)
        self.yaw_enu = None            # EV heading [rad, ENU]
        self.wp_i = 0
        self.last_cmd_t = 0.0
        self.last_land_t = 0.0
        self.last_origin_t = 0.0
        self.origin_count = 0
        self.disarmed = False
        self.aborted = False
        self.last_yaw_cmd = YAW
        self.last_state = None

        be = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status_v1',
                                 self.on_status, be)
        self.create_subscription(VehicleLocalPosition,
                                 '/fmu/out/vehicle_local_position_v1',
                                 self.on_lpos, be)
        self.create_subscription(VehicleLandDetected,
                                 '/fmu/out/vehicle_land_detected',
                                 self.on_landed, be)
        self.create_subscription(Odometry, '/mcl/odom', self.on_ev, 10)
        self.create_subscription(String, '/mcl/health', self.on_health, 10)

        self.ev_pub = self.create_publisher(
            VehicleOdometry, '/fmu/in/vehicle_visual_odometry', 10)
        self.ocm_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', 10)
        self.sp_pub = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', 10)
        self.cmd_pub = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', 10)
        self.state_pub = self.create_publisher(String, 'mission/state', 10)

        self.create_timer(1.0 / SETPOINT_HZ, self.tick)
        self.create_timer(1.0, self.publish_state)
        self.get_logger().info(
            'mission manager up: mode=%s ev_frame=%s cruise_alt=%.1f m '
            'margin=%.0f m (waiting for fmu + /mcl/odom)'
            % (self.mission_mode, self.ev_frame, self.guard.cruise_alt,
               self.guard.margin_m))

    # ------------------------------------------------------------------ subs
    def on_status(self, msg):
        self.status = msg
        self.status_t = self.get_clock().now().nanoseconds * 1e-9

    def on_lpos(self, msg):
        self.lpos = msg

    def on_landed(self, msg):
        self.landed = bool(msg.landed)

    def on_health(self, msg):
        try:
            self.health = json.loads(msg.data)
            self.health_t = self.get_clock().now().nanoseconds * 1e-9
        except ValueError:
            self.get_logger().warn('bad /mcl/health JSON', once=True)

    def on_ev(self, msg):
        """EV bridge: nav_msgs/Odometry (ENU/FLU) -> VehicleOdometry (NED/FRD).

        Honest pose -> position in the PX4 local frame (see ev_frame param);
        gated pose -> NaN position (EKF2 then skips position fusion but still
        fuses the honest twist). Twist is always honest (body FLU -> FRD).
        """
        p = msg.pose.pose.position
        pc = msg.pose.covariance
        tc = msg.twist.covariance
        honest = pc[0] < POSE_GATED_COV and pc[7] < POSE_GATED_COV

        out = VehicleOdometry()
        now_us = self.get_clock().now().nanoseconds // 1000
        out.timestamp = now_us
        out.timestamp_sample = now_us
        out.pose_frame = VehicleOdometry.POSE_FRAME_NED
        out.velocity_frame = VehicleOdometry.VELOCITY_FRAME_BODY_FRD

        if honest:
            if self.ev_frame == 'first_fix' and self.p_fix is None:
                self.p_fix = (p.x, p.y)
                self.get_logger().info(
                    'EV lock: map fix (%.1f, %.1f) -> local origin'
                    % (p.x, p.y))
            if self.ev_frame == 'first_fix':
                ox, oy = self.p_fix
            else:
                ox, oy = self.local_origin_map_x, self.local_origin_map_y
            # ENU (x,y,z) -> NED (y,x,-z), minus the local-frame origin
            out.position = [p.y - oy, p.x - ox, -p.z]
            out.position_variance = [pc[7], pc[0], pc[14]]
            q = msg.pose.pose.orientation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                             1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            self.yaw_enu = yaw
            w, x, y, z = qmul(Q_ENU2NED,
                              qmul((q.w, q.x, q.y, q.z), Q_FLU2FRD))
            out.q = [w, x, y, z]
            out.orientation_variance = [pc[21], pc[28], pc[35]]
            self.ev_map_pos = (p.x, p.y)      # map ENU for the guard
        else:
            out.position = [float('nan')] * 3
            out.position_variance = [pc[7], pc[0], pc[14]]
            out.q = [float('nan')] * 4
            out.orientation_variance = [pc[21], pc[28], pc[35]]
            self.ev_map_pos = None

        v = msg.twist.twist.linear
        vz = 0.0 if self.ev_vz_zero else -v.z     # sim: tilt-coupled bag z
        out.velocity = [v.x, -v.y, vz]            # body FLU -> body FRD
        out.angular_velocity = [float('nan')] * 3
        out.velocity_variance = [tc[0], tc[7], tc[14]]
        self.ev_pub.publish(out)

        self.ev = msg
        self.ev_t = self.get_clock().now().nanoseconds * 1e-9
        if self.first_ev_t is None:
            self.first_ev_t = self.ev_t
            self.get_logger().info('EV stream started (waiting to arm)')

    # --------------------------------------------------------------- helpers
    def send_command(self, command, p1=0.0, p2=0.0, p3=0.0, p4=0.0,
                     p5=0.0, p6=0.0, p7=0.0):
        c = VehicleCommand()
        c.timestamp = self.get_clock().now().nanoseconds // 1000
        c.param1, c.param2, c.param3, c.param4 = p1, p2, p3, p4
        c.param5, c.param6, c.param7 = p5, p6, p7
        c.command = command
        c.target_system = 1
        c.target_component = 1
        c.source_system = 1
        c.source_component = 1
        c.from_external = True
        self.cmd_pub.publish(c)

    def mission_done(self, now, ev_age):
        if self.mission_mode == 'waypoints':
            return self.wp_i >= len(self.waypoints)
        return (self.first_ev_t is not None and ev_age is not None
                and ev_age > STREAM_END_S
                and now - self.first_ev_t > 10.0)

    def alt(self):
        return None if self.lpos is None else -self.lpos.z

    def feedforward_velocity(self, ev_age):
        """(v_ned, yaw_ned) shadowing the EV stream, or None when stale."""
        if self.ev is None or ev_age is None or ev_age > 0.5:
            return None
        v = self.ev.twist.twist.linear
        if self.yaw_enu is not None:
            th = math.pi / 2.0 - self.yaw_enu      # EV ENU yaw -> NED yaw
            yaw = th
        else:
            th, yaw = 0.0, YAW
        c, s = math.cos(th), math.sin(th)
        north = v.x * c + v.y * s                  # body FLU -> NED horizontal
        east = v.x * s - v.y * c
        vz = -v.z                                  # FLU up -> NED down
        alt = self.alt()
        if self.ff_altitude_hold and alt is not None:
            vz = -max(-ALT_VZ_MAX,
                      min(ALT_VZ_MAX, ALT_KP * (self.guard.cruise_alt - alt)))
        return ([max(-self.v_max, min(self.v_max, north)),
                 max(-self.v_max, min(self.v_max, east)),
                 max(-self.v_max, min(self.v_max, vz))], yaw)

    def waypoint_velocity(self):
        """(v_ned, yaw_ned) toward the active waypoint, or None without a fix."""
        if self.ev_map_pos is None:
            return None
        while self.wp_i < len(self.waypoints):
            wx, wy = self.waypoints[self.wp_i]
            dx, dy = wx - self.ev_map_pos[0], wy - self.ev_map_pos[1]
            dist = math.hypot(dx, dy)
            if dist > WP_RADIUS_M:
                break
            self.get_logger().info('waypoint %d reached (%.1f m)'
                                   % (self.wp_i, dist))
            self.wp_i += 1
        if self.wp_i >= len(self.waypoints):
            return None                     # mission done -> guard lands
        wx, wy = self.waypoints[self.wp_i]
        dx, dy = wx - self.ev_map_pos[0], wy - self.ev_map_pos[1]
        dist = max(1e-3, math.hypot(dx, dy))
        sp = min(self.v_max, 0.7 * dist)    # P-controller, clamped
        v_e, v_n = sp * dx / dist, sp * dy / dist
        vz = 0.0
        alt = self.alt()
        if alt is not None:
            vz = -max(-ALT_VZ_MAX,
                      min(ALT_VZ_MAX, ALT_KP * (self.guard.cruise_alt - alt)))
        v_ned = [v_n, v_e, vz]
        return v_ned, math.atan2(v_e, v_n)  # NED yaw = atan2(east, north)

    def return_velocity(self):
        """Fly back toward the map centre, or None without a fix."""
        if self.ev_map_pos is None:
            return None
        x, y = self.ev_map_pos               # ENU: x east, y north
        dist = max(1e-3, math.hypot(x, y))
        v_e, v_n = -self.v_max * x / dist, -self.v_max * y / dist
        return ([v_n, v_e, 0.0], math.atan2(v_e, v_n))

    # ----------------------------------------------------------------- tick
    def tick(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        now_us = self.get_clock().now().nanoseconds // 1000
        if self.aborted or self.status is None or now - self.status_t > 3.0:
            return                            # fmu not alive yet
        if self.health is not None and now - self.health_t > 5.0:
            self.health = None                # health channel stalled -> EV only

        # EKF2 global origin = map centre (repeat during the first seconds)
        if (self.set_global_origin and self.origin_count < 6
                and now - self.last_origin_t > 2.0):
            self.send_command(CMD_SET_GPS_GLOBAL_ORIGIN,
                              p5=self.map_center_lat, p6=self.map_center_lon)
            self.last_origin_t = now
            self.origin_count += 1

        # OFFBOARD stream: control mode + setpoint (velocity + yaw), 50 Hz
        ocm = OffboardControlMode()
        ocm.timestamp = now_us
        ocm.velocity = True
        self.ocm_pub.publish(ocm)

        ev_age = None if self.ev_t is None else now - self.ev_t
        ready = (self.status.nav_state == NAV_STATE_OFFBOARD
                 and self.status.arming_state == ARMING_STATE_ARMED)
        st = self.guard.update(now, ready, self.health, ev_age,
                               self.ev_map_pos, self.alt(),
                               self.mission_done(now, ev_age), self.landed)
        if st != self.last_state:
            self.get_logger().info('guard: %s -> %s (%s)'
                                   % (STATE_NAMES[self.last_state]
                                      if self.last_state is not None else '-',
                                      STATE_NAMES[st], self.guard.reason))
            self.last_state = st

        # --- setpoint for the state -------------------------------------
        v_ned, yaw = [0.0, 0.0, 0.0], self.last_yaw_cmd
        if st == TAKEOFF:
            v_ned = [0.0, 0.0, -CLIMB_RATE]
        elif st == MISSION and self.mission_mode == 'feedforward':
            ff = self.feedforward_velocity(ev_age)
            if ff is not None:
                v_ned, yaw = ff
        elif st == MISSION:
            wp = self.waypoint_velocity()
            if wp is not None:
                v_ned, yaw = wp
        elif st == RETURN:
            rt = self.return_velocity()
            if rt is not None:
                v_ned, yaw = rt
        if not math.isnan(yaw):
            self.last_yaw_cmd = yaw
        sp = TrajectorySetpoint()
        sp.timestamp = now_us
        nan = float('nan')
        sp.position = [nan, nan, nan]
        sp.acceleration = [nan, nan, nan]
        sp.jerk = [nan, nan, nan]
        sp.velocity = v_ned
        sp.yawspeed = 0.0
        sp.yaw = yaw
        self.sp_pub.publish(sp)

        # --- commands ----------------------------------------------------
        # Keep the vehicle in OFFBOARD for the WHOLE mission: a PX4 failsafe
        # (RTL/land) drops out of OFFBOARD, and with the mode command only in
        # the preflight states the manager would keep commanding a vehicle
        # that no longer listens. ARM stays pre-takeoff only — a disarm later
        # means PX4 landed, which the guard's LAND -> DONE path handles.
        # NB: never gate this on the PX4 land detector — it is True while the
        # vehicle sits on the ground before takeoff (self.landed only feeds
        # the guard).
        if st in (WAIT, TAKEOFF, MISSION, HOLD, RETURN):
            if now - self.last_cmd_t > 1.0:
                self.last_cmd_t = now
                if self.status.nav_state != NAV_STATE_OFFBOARD:
                    if (self.first_ev_t is not None
                            and now - self.first_ev_t > 2.0):
                        self.send_command(CMD_DO_SET_MODE, *MODE_OFFBOARD)
                elif (st in (WAIT, TAKEOFF)
                      and self.status.arming_state != ARMING_STATE_ARMED):
                    self.send_command(CMD_ARM_DISARM, p1=1.0)
        if st == LAND and now - self.last_land_t > 2.0:
            if self.status.nav_state != NAV_STATE_AUTO_LAND:
                self.last_land_t = now
                self.send_command(CMD_DO_SET_MODE, *MODE_AUTO_LAND)
        if st == DONE and not self.disarmed:
            if self.status.arming_state == ARMING_STATE_ARMED:
                self.send_command(CMD_ARM_DISARM, p1=0.0)
            else:
                self.disarmed = True
                self.get_logger().info('mission complete: landed + disarmed')

        # preflight abort: never armed within the budget -> leave on the ground
        if (st == WAIT and self.first_ev_t is not None
                and now - self.first_ev_t > self.preflight_max_s):
            self.get_logger().error(
                'preflight timeout: not armed after %.0f s of EV — aborting '
                '(no flight)' % self.preflight_max_s)
            self.aborted = True
            rclpy.shutdown()

    # ---------------------------------------------------------------- state
    def publish_state(self):
        h = (self.health or {}).get('state') if self.health else None
        alt = self.alt()
        msg = {
            't': round(time.time(), 3),
            'state': self.guard.state_name,
            'reason': self.guard.reason,
            'mode': self.mission_mode,
            'health': h,
            'ev_age': (None if self.ev_t is None else
                       round(self.get_clock().now().nanoseconds * 1e-9
                             - self.ev_t, 2)),
            'hold_total': round(self.guard.hold_total, 1),
            'wp': self.wp_i,
            'pos': (None if self.ev_map_pos is None
                    else [round(self.ev_map_pos[0], 1),
                          round(self.ev_map_pos[1], 1)]),
            'alt': None if alt is None else round(alt, 1),
            'armed': (self.status is not None
                      and self.status.arming_state == ARMING_STATE_ARMED),
            'nav_state': None if self.status is None else self.status.nav_state,
            'landed': self.landed,
        }
        self.state_pub.publish(String(data=json.dumps(msg)))


def main():
    rclpy.init()
    node = MissionManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()