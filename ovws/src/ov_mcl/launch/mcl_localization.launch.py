"""Online GNSS-denied localization: SchurVINS (OpenVINS) + MCL node.

Starts the VIO subscribe node (defaults to the drone's
cyperstereo_012_752x480_equi fisheye calibration, used by the 20260831
bags) and the ov_mcl particle-filter node. Optionally also plays a
rosbag2 (for replay testing).

Usage:
  ros2 launch ov_mcl mcl_localization.launch.py
  ros2 launch ov_mcl mcl_localization.launch.py bag:=/path/to/bag_dir
  ros2 launch ov_mcl mcl_localization.launch.py vio_config:=cyperstereo_c76_8mm_752x480
"""

import os

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess,
                            IncludeLaunchDescription)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    mcl_share = get_package_share_directory('ov_mcl')
    ov_share = get_package_share_directory('ov_msckf')

    bag_arg = LaunchConfiguration('bag')
    default_config = 'cyperstereo_012_752x480_equi'
    default_calib = os.path.join(
        ov_share, 'config', default_config, 'kalibr_imucam_chain.yaml')

    # Portable defaults: the first root that actually holds mcl.py wins —
    # $MCL_ROOT, four levels up (…/MCLDrone/ovws/src/ov_mcl/launch/), the dev
    # checkout, then ~/MCLDrone (deployed layout: package installed from ~/ovws).
    _up = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       '..', '..', '..', '..'))
    _root = next((p for p in (os.environ.get('MCL_ROOT'), _up,
                              '/home/one/GNSS-denied-Localization',
                              os.path.expanduser('~/MCLDrone'))
                  if p and os.path.isfile(os.path.join(p, 'mcl.py'))), _up)
    # bundle marker: the MCLDrone maps/ layout (absent in the dev checkout)
    _bundle = os.path.exists(os.path.join(_root, 'maps', 'z17_5120.png'))

    def _portable(rel, dev):
        return os.path.join(_root, rel) if _bundle else dev

    map_default = _portable(
        'maps/z17_5120.png',
        '/home/one/GNSS-denied-Localization/z17_5120.png')
    ckpt_default = _portable(
        'checkpoints_huizhou/huizhou_ft_mt_1_epoch_040_of_40.pt',
        '/home/one/GNSS-denied-Localization/checkpoints_huizhou/'
        'huizhou_ft_mt_1_epoch_040_of_40.pt')
    out_default = os.path.join(
        _portable('mcl_runs',
                  '/home/one/GNSS-denied-Localization/sample_images/mcl_test'),
        'online')

    args = [
        DeclareLaunchArgument(
            'vio_config', default_value=default_config,
            description='VIO config name (in ov_msckf/config/) or a '
                        'full path to an estimator_config.yaml'),
        DeclareLaunchArgument(
            'calib_path', default_value=default_calib,
            description='kalibr_imucam_chain.yaml for the MCL node '
                        '(undistortion + T_cam_imu); keep in sync with '
                        'vio_config'),
        DeclareLaunchArgument(
            'bag', default_value='',
            description='optional rosbag2 directory to play (replay test)'),
        DeclareLaunchArgument('map_path', default_value=map_default),
        DeclareLaunchArgument(
            'checkpoint',
            default_value=ckpt_default,
            description='similarity model checkpoint (mcl.py CHECKPOINT); '
                        'use the FINE-TUNED ckpt — the pretrain-only ckpt '
                        'misses truth and false-locks'),
        DeclareLaunchArgument('out_dir', default_value=out_default),
        DeclareLaunchArgument('stride', default_value='5'),
        DeclareLaunchArgument(
            'frame_start', default_value='0',
            description='first camera frame to process (0 = from the start); '
                        'skips the takeoff stage'),
        DeclareLaunchArgument(
            'frame_end', default_value='0',
            description='one past the last camera frame to process '
                        '(0 = to the end); skips the landing stage'),
        DeclareLaunchArgument('n_particles', default_value='1000'),
        DeclareLaunchArgument('init_mode', default_value='scan'),
        DeclareLaunchArgument('init_px_x', default_value='0',
                              description='init point map px (init_mode=point)'),
        DeclareLaunchArgument('init_px_y', default_value='0',
                              description='init point map py (init_mode=point)'),
        DeclareLaunchArgument(
            'init_lat', default_value='-999.0',
            description='init WGS-84 latitude deg (init_mode=point; '
                        'overrides init_px_x/y)'),
        DeclareLaunchArgument(
            'init_lon', default_value='-999.0',
            description='init WGS-84 longitude deg (init_mode=point; '
                        'overrides init_px_x/y)'),
        DeclareLaunchArgument(
            'init_radius_m', default_value='300.0',
            description='init_mode=point prior uncertainty radius [m]: the '
                        'init scan sweeps this region around the clicked '
                        'point (min 48, max ~2000; 300 m = typical click '
                        'uncertainty on a 5 km map)'),
        DeclareLaunchArgument(
            'yaw_cal', default_value='-1,1.3',
            description='compass->MCL-map yaw calibration "a,b" (b in deg): '
                        'yaw_mcl = a*fcu_ned_heading + radians(b). Fitted on '
                        'bag 190020 s2 replay (residual std 4.3 deg); '
                        'a=-1 (NED heading is CW-positive), b = declination '
                        '+ camera-mounting offset — rig+map constant'),
        DeclareLaunchArgument(
            'map_geo_offset_x', default_value='-27449088.0',
            description='map georeference: local px = global z17 mercator px '
                        '+ offset (default matches z17_5120.png)'),
        DeclareLaunchArgument(
            'map_geo_offset_y', default_value='-14586624.0',
            description='map georeference y (see map_geo_offset_x)'),
        DeclareLaunchArgument('alt_anchor', default_value='0.0'),
        DeclareLaunchArgument(
            'fcu_alt_topic', default_value='/mavros/local_position/pose',
            description='live FCU baro height + EKF attitude source (MAVROS '
                        'ENU local position pose); empty disables. Replay '
                        'reads the ULog beside the bag instead'),
        DeclareLaunchArgument(
            'fcu_imu_rot', default_value='',
            description='FCU-body(FRD) -> /imu0 mounting rotation, 9 '
                        'row-major values comma-separated (empty = built-in '
                        'Wahba-fitted default from bag 190020; recalibrate '
                        'if the rig is re-mounted)'),
        DeclareLaunchArgument(
            'save_debug', default_value='false',
            description='save per-frame pipeline images to out_dir/debug'),
        DeclareLaunchArgument(
            'debug_stride', default_value='1',
            description='save debug images every N-th processed frame '
                        '(1 = every processed frame; with stride=5 that is '
                        'already every 5th camera frame)'),
        DeclareLaunchArgument(
            'ev_topic', default_value='/mcl/odom',
            description='EV odometry output topic (PX4 EKF2 External Vision, '
                        'the GPS substitute). Remap to /mavros/odometry/out '
                        'when MAVROS runs; keep the default for bag-replay '
                        'inspection'),
    ]

    vio = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ov_share, 'launch', 'subscribe.launch.py')),
        launch_arguments={'config': LaunchConfiguration('vio_config')}.items())

    mcl = Node(
        package='ov_mcl',
        executable='mcl_node',
        output='screen',
        parameters=[{
            'map_path': LaunchConfiguration('map_path'),
            'checkpoint': LaunchConfiguration('checkpoint'),
            'out_dir': LaunchConfiguration('out_dir'),
            'calib_path': LaunchConfiguration('calib_path'),
            'stride': LaunchConfiguration('stride'),
            'frame_start': LaunchConfiguration('frame_start'),
            'frame_end': LaunchConfiguration('frame_end'),
            'n_particles': LaunchConfiguration('n_particles'),
            'init_mode': LaunchConfiguration('init_mode'),
            'init_px_x': LaunchConfiguration('init_px_x'),
            'init_px_y': LaunchConfiguration('init_px_y'),
            'init_lat': LaunchConfiguration('init_lat'),
            'init_lon': LaunchConfiguration('init_lon'),
            'init_radius_m': LaunchConfiguration('init_radius_m'),
            'yaw_cal': LaunchConfiguration('yaw_cal'),
            'map_geo_offset_x': LaunchConfiguration('map_geo_offset_x'),
            'map_geo_offset_y': LaunchConfiguration('map_geo_offset_y'),
            'alt_anchor': LaunchConfiguration('alt_anchor'),
            'fcu_alt_topic': LaunchConfiguration('fcu_alt_topic'),
            'fcu_imu_rot': LaunchConfiguration('fcu_imu_rot'),
            'save_debug': LaunchConfiguration('save_debug'),
            'debug_stride': LaunchConfiguration('debug_stride'),
            # bag dir is passed to the node ONLY for the replay.png GPS-truth
            # overlay (the filter itself never sees it — GNSS-denied)
            'bag': bag_arg,
        }],
        remappings=[('mcl/odom', LaunchConfiguration('ev_topic'))],
    )

    bag_play = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', bag_arg, '--delay', '3'],
        condition=IfCondition(
            PythonExpression(["'", bag_arg, "' != ''"])),
        output='screen',
    )

    return LaunchDescription(args + [bag_play, vio, mcl])
