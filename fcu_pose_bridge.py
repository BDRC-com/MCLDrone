#!/usr/bin/env python3
"""PX4 uXRCE-DDS -> MCL FCU-pose bridge (MAVROS replacement).

The MCL node consumes ONE FCU input: a geometry_msgs/PoseStamped carrying
the flight-controller EKF height + attitude (its /mavros/local_position/
pose contract on MAVROS rigs). On the pure-PX4 vehicle those facts arrive
natively over uXRCE-DDS as:
  /fmu/out/vehicle_local_position_v1  px4_msgs/VehicleLocalPosition  (~50 Hz)
  /fmu/out/vehicle_attitude           px4_msgs/VehicleAttitude       (~100 Hz)
This node merges them and republishes in the ROS REP-103 convention MCL
already expects:
  position: PX4 local NED (x=N, y=E, z=D) -> ENU (x=E, y=N, z=U)
            x_enu = y_ned, y_enu = x_ned, z_enu = -z_ned
  attitude: q rotates FCU body FRD -> NED world (w,x,y,z); MCL needs the
            base_link FLU -> ENU world quaternion (ROS x,y,z,w):
              R_enu = P R_ned D,  P: NED coords -> ENU coords
                                    [[0,1,0],[1,0,0],[0,0,-1]],
                                    D: FLU coords -> FRD coords = Rx(pi)
            (the naive single 180-deg-about-x conjugation that the
            MAVROS source shorthand suggests maps a north yaw onto east
            and is WRONG here; the P R D product is the unique physical
            transform — verified to recover the PX4 heading field and
            zero level-hover tilt on the flight rig, 2026-09-23).
Stamps use the COMPANION clock (not the FCU boot clock) so MCL's
stamp-matched interpolation lines up with the camera header stamps; the
UDP latency at 50-100 Hz is negligible against its 0.5-1 s gates.

Run (every terminal: ROS_DOMAIN_ID=0 + source ~/my_px4/install/setup.zsh):
  python3 fcu_pose_bridge.py
  ros2 launch ov_mcl mcl_localization.launch.py \
      fcu_alt_topic:=/fcu/local_position/pose ...
"""
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from px4_msgs.msg import VehicleLocalPosition, VehicleAttitude
from geometry_msgs.msg import PoseStamped

A = math.sqrt(0.5)


def qmul(q1, q2):
    """Hamilton product of two w-first quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + w2 * x1 + y1 * z2 - z1 * y2,
            w1 * y2 + w2 * y1 + z1 * x2 - x1 * z2,
            w1 * z2 + w2 * z1 + x1 * y2 - y1 * x2)


# q_P: quaternion of the NED-coords -> ENU-coords permutation P
_Q_P = (0.0, A, A, 0.0)
# q_D: 180 deg about x (FLU coords -> FRD coords)
_Q_D = (0.0, 1.0, 0.0, 0.0)


def ned_frd_to_enu_flu(q):
    """(w,x,y,z) FRD->NED quaternion -> (w,x,y,z) FLU->ENU quaternion."""
    return qmul(_Q_P, qmul(q, _Q_D))


class FcuPoseBridge(Node):
    def __init__(self):
        super().__init__('fcu_pose_bridge')
        self.declare_parameter('lpos_topic',
                               '/fmu/out/vehicle_local_position_v1')
        self.declare_parameter('att_topic', '/fmu/out/vehicle_attitude')
        self.declare_parameter('out_topic', '/fcu/local_position/pose')
        self.declare_parameter('frame_id', 'map')

        self.att = None                 # latest (w,x,y,z)
        self.z_ok = False
        be = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(VehicleLocalPosition,
                                 str(self.get_parameter('lpos_topic').value),
                                 self.on_lpos, be)
        self.create_subscription(VehicleAttitude,
                                 str(self.get_parameter('att_topic').value),
                                 self.on_att, be)
        self.pub = self.create_publisher(
            PoseStamped, str(self.get_parameter('out_topic').value), 10)
        self.get_logger().info(
            'fcu_pose_bridge up: %s + %s -> %s (NED/FRD -> ENU/FLU)'
            % (self.get_parameter('lpos_topic').value,
               self.get_parameter('att_topic').value,
               self.get_parameter('out_topic').value))

    def on_att(self, msg):
        self.att = (float(msg.q[0]), float(msg.q[1]),
                    float(msg.q[2]), float(msg.q[3]))

    def on_lpos(self, msg):
        if not msg.z_valid:
            return                       # MCL needs an honest baro/EKF height
        if self.att is None:
            return                       # wait for the first attitude sample
        now = self.get_clock().now().to_msg()
        out = PoseStamped()
        out.header.stamp = now
        out.header.frame_id = str(self.get_parameter('frame_id').value)
        # NED -> ENU position
        out.pose.position.x = float(msg.y)     # east
        out.pose.position.y = float(msg.x)     # north
        out.pose.position.z = -float(msg.z)    # up
        # FRD->NED -> FLU->ENU attitude
        w, x, y, z = ned_frd_to_enu_flu(self.att)
        out.pose.orientation.x = x
        out.pose.orientation.y = y
        out.pose.orientation.z = z
        out.pose.orientation.w = w
        self.pub.publish(out)
        if not self.z_ok:
            self.z_ok = True
            self.get_logger().info(
                'first valid FCU pose: z=%.1f m AGL, xy_valid=%s '
                '(bridge live)' % (-float(msg.z), msg.xy_valid))


def main():
    rclpy.init()
    rclpy.spin(FcuPoseBridge())


if __name__ == '__main__':
    main()
