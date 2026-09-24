# 配置环境
```bash
git clone https://github.com/Mastopke304/MCLDrone.git
cd MCLDrone

# Conda 训练环境
conda env create -f environment.yml
# MCL 运行环境
# 需要先安装ROS2 Humble 和 MicroXRCE-DDS agent，然后：
sudo apt install -y python3-opencv python3-matplotlib python3-scipy python3-pandas \
  python3-yaml python3-pip python3-setuptools python3-colcon-common-extensions
# 然后安装环境：
source /opt/ros/humble/setup.bash
sudo /usr/bin/python3 -m pip install -r requirements.txt
/usr/bin/python3 - <<'EOF'
import torch, numpy as np, cv2
assert torch.cuda.is_available(), "torch must see CUDA (JetPack wheel)"
print(torch.__version__, "cuda", torch.version.cuda, "numpy", np.__version__)
EOF
# 应出现: 2.5.0a0+...nv24.08 cuda 12.6 numpy 1.26.x
```
如果`pip`因为 PEP-668 失败，需要添加`--break-system-packages`参数；不要让它升级`numpy`到2.x（检查安装后是否成功）。

创建好环境后，需要将如下文件放置在制定地点：
```bash
# 模型权重 huizhou_ft_mt_1_epoch_040_of_40.pt
cd checkpoints_huizhou/
# 地图 z17_5120.png
cd maps/
# ov_SchurVINS
cd ovws/src/
git clone https://github.com/BDRC-com/ov_SchurVINS.git
# Build
cd ~/MCLDrone/ovws/
colcon build
```
# 在ROS2 Bag中运行MCLDrone
## 1. 运行ov_SchurVINS + MCL -> Odometry
```bash
cd ~/MCLDrone
log_out_dir=./test/mcl_replay_190020_log
bag_out_dir=./test/mcl_replay_190020_bag
mkdir -p $log_out_dir $bag_out_dir
source /opt/ros/$ROS2_DISTRO/setup.bash
source ~/MCLDrone/ovws/install/setup.bash
# 启动ov_SchurVINS + MCL -> Odometry
ros2 launch ov_mcl mcl_localization.launch.py \
  bag:=/your/path/to/bag_0001_20260831_190020 \
  init_mode:=point init_lat:=22.842897 init_lon:=114.525573 init_radius_m:=300.0 \
  map_geo_offset_x:=-27449088.0 map_geo_offset_y:=-14586624.0 \
  out_dir:=$log_out_dir
# 录制Odometry
ros2 bag record /mcl/odom /mcl/health /ov_msckf/odomimu -o $bag_out_dir
```
会在`$bag_out_dir`目录下生成bag文件`mcl_replay_190020_bag_0.db3`，在`$log_out_dir`目录下生成测试结果。

可以通过如下命令查看VINS：
```bash
rviz2 -d ~/MCLDrone/ovws/src/ov_SchurVINS/ov_msckf/launch/display_ros2.rviz
```

## 2. 可视化Odometry和GPS Truth
```bash
python3 ~/MCLDrone/plot_odom_vs_gps.py \
    $log_out_dir \  # MCL run directory
    --bag $bag_out_dir \  # Odometry records
    --fly ~/datasets/bag_0001_20260831_184004  # GPS truth
```
会在`$log_out_dir`目录下生成可视化结果。

# 在线运行MCLDrone并录制包
## 1. 启动摄像机
```bash
ssh ros2@192.168.1200
export ROS_DOMAIN_ID=0
source /opt/ros/humble/setup.zsh
source ~/MCLDrone/ovws/install/setup.zsh
source ~/my_px4/install/setup.zsh
# 后台启动摄像机
nohup ~/start_camera.sh > /tmp/camera_bridge.log 2>&1 & disown
ros2 topic hz /imu0 # 检查IMU话题频率
```

## 2. 启动MCLDrone
可选启动方式：
 - 使用脚本启动MCLDrone（跳转2-1）
 - 手动启动MCLDrone（跳转2-2）

### 2-1. 启动MCLDrone（脚本）
#### 2-1-1. 使用脚本启动MCLDrone
```bash
~/MCLDrone/run_live.sh --name flight01 --daemon \ # --name 可以是任何你想要的名字
  --map ~/MCLDrone/maps/z17_5120.png \
  --lat 22.842897 --lon 114.525573 # 起飞点的经纬度
```

#### 2-1-2. 重新连接后检查
```bash
~/MCLDrone/run_live.sh --name flight01 --status
# Or
tail -f /tmp/run_live_flight01.console.log
```

#### 2-1-3. 结束录制
```bash
~/MCLDrone/run_live.sh --name flight01 --stop
```

#### 2-1-4. 关闭摄像机
```bash
pgrep -af 'cyperstereo|capture_image|mjpeg' # 检查是否有进程在运行
pkill -INT -f 'ros2 launch cyperstereo_ros2_bridge' # 关闭进程
pgrep -af 'cyperstereo|capture_image|mjpeg' # 检查进程是否已关闭（应该为空）
```

### 2-2. 启动MCLDrone（手动启动）
#### 2-2-1. 启动MicroXRCE-DDS agent
```bash
export ROS_DOMAIN_ID=0
source /opt/ros/humble/setup.zsh
source ~/MCLDrone/ovws/install/setup.zsh
source ~/my_px4/install/setup.zsh
NAME=flight01 # 你可以选择任何你想要的名字
mkdir -p ~/ros2bag/mcl_runs ~/ros2bag/${NAME}_flightlog
# MicroXRCE-DDS agent
nohup ~/my_px4/install/microxrcedds_agent/bin/MicroXRCEAgent udp4 -p 8888 -d 0 \
  > ~/ros2bag/${NAME}_flightlog/agent.log 2>&1 &
disown
ros2 topic list # 检查Agent是否启动，应有许多fmu话题
```
注意别启动了多个MicroXRCE-DDS agent进程，用`ss -ulnp | grep 8888`检查。



#### 2-2-2. 启动FCU Pose Bridge
```bash
cd ~/MCLDrone
nohup env FASTRTPS_DEFAULT_PROFILES_FILE=~/MCLDrone/fastdds_udp_only.xml \
  python3 fcu_pose_bridge.py > ~/ros2bag/${NAME}_flightlog/bridge.log 2>&1 &
disown
ros2 topic hz /fcu/local_position/pose # 检查FCU Pose Bridge话题频率
```

#### 2-2-3. 启动Recorder
启动两个Recorder，一个记录图像，一个记录数据：
```bash
# images (heavy) — default SHM transport
nohup ros2 bag record -s mcap -o ~/ros2bag/${NAME}_img \
  /cam0/image_raw /cam1/image_raw \
  > ~/ros2bag/${NAME}_flightlog/rec_img.log 2>&1 &
disown

# high-rate small messages — UDP-only profile (or /imu0 records at ~25 Hz)
nohup env FASTRTPS_DEFAULT_PROFILES_FILE=~/MCLDrone/fastdds_udp_only.xml \
  ros2 bag record -s mcap -o ~/ros2bag/${NAME}_data \
    /imu0 /ov_msckf/odomimu /mcl/odom /mcl/health \
    /fcu/local_position/pose \
    /fmu/out/vehicle_local_position_v1 /fmu/out/vehicle_attitude \
    /fmu/out/vehicle_status_v1 /fmu/in/vehicle_visual_odometry \
  > ~/ros2bag/${NAME}_flightlog/rec_data.log 2>&1 &
disown
tail -f ~/ros2bag/${NAME}_flightlog/rec_data.log ## 检查，可选
```

#### 2-2-4. 启动VIO + MCL -> Odometry
```bash
nohup ros2 launch ov_mcl mcl_localization.launch.py \
  map_path:=/home/ros2/MCLDrone/maps/z17_5120.png \
  map_geo_offset_x:=-27449088.0 map_geo_offset_y:=-14586624.0 \
  init_mode:=point init_lat:=22.842897 init_lon:=114.525573 \
  init_radius_m:=300.0 stride:=5 yaw_cal:=-1,1.3 \
  fcu_alt_topic:=/fcu/local_position/pose \
  out_dir:=/home/ros2/ros2bag/mcl_runs/${NAME} \
  > ~/ros2bag/${NAME}_flightlog/mcl.log 2>&1 &
disown

# 重新连接后检查
tail -f ~/ros2bag/${NAME}_flightlog/mcl.log
# expect: SchurVINS init -> "MCL node ready" -> during climb:
#   point init waiting for coverage  -> recovery CONFIRMED -> tracking
ros2 topic echo /mcl/health --once
ros2 topic hz /mcl/odom
```

#### 2-2-5. 结束录制（按顺序操作）
```bash
# 1. 停止MCL node (SIGINT DIRECTLY)
pkill -INT -f '/lib/ov_mcl/mcl_node'
for i in $(seq 1 40); do pgrep -f '/lib/ov_mcl/mcl_node' >/dev/null || break; sleep 1; done
grep -E 'saved .*replay_log.npz|no frames processed' ~/ros2bag/${NAME}_flightlog/mcl.log | tail -2
pkill -INT -f 'ros2 launch ov_mcl' 2>/dev/null; sleep 2

# 2. 停止Recorder (SIGINT DIRECTLY)
pkill -INT -f 'ros2 bag record'
for i in $(seq 1 15); do pgrep -f 'ros2 bag record' >/dev/null || break; sleep 1; done


# 3. 停止其他进程
pkill -f run_subscribe_msckf
pkill -INT -f fcu_pose_bridge.py; sleep 1
pkill -INT -f MicroXRCEAgent

# 4. 验证数据包
ros2 bag info ~/ros2bag/${NAME}_data/*_0.mcap | \
  grep -E 'imu0|fcu/local_position|odomimu'
ls -la ~/ros2bag/mcl_runs/${NAME}/
```

## Extra Tips
```bash
# 杀掉所有进程
pkill -f MicroXRCEAgent
pkill -f fcu_pose_bridge.py
pkill -f 'ros2 bag record'
pkill -f '/lib/ov_mcl/mcl_node'
pkill -f run_subscribe_msckf
ss -ulnp | grep 8888 # 需要为空
```