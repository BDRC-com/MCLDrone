# 在ROS2 Bag中运行MCLDrone
## 1. 配置环境
```bash
git clone https://github.com/Mastopke304/MCLDrone.git
cd MCLDrone

# Conda 训练环境
conda env create -f environment_sivl.yml
# MCL 运行环境
# 需要先安装ROS2 Humble 和 MicroXRCE-DDS agent，然后：
sudo apt install -y python3-opencv python3-matplotlib python3-scipy python3-pandas \
  python3-yaml python3-pip python3-setuptools python3-colcon-common-extensions
# 然后安装环境：
source /opt/ros/humble/setup.bash
sudo /usr/bin/python3 -m pip install -r ~/MCLDrone/requirements_drone.txt
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

## 2. 运行ov_SchurVINS + MCL -> Odometry
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

## 3. 可视化Odometry和GPS Truth
```bash
python3 ~/MCLDrone/plot_odom_vs_gps.py \
    $log_out_dir \  # MCL run directory
    --bag $bag_out_dir \  # Odometry records
    --fly ~/datasets/bag_0001_20260831_184004  # GPS truth
```
会在`$log_out_dir`目录下生成可视化结果。

# 在线运行MCLDrone并录制包
## 1. 启动摄像机（在终端 1中）
```bash
# 终端 1
ssh ros2@192.168.1200
export ROS_DOMAIN_ID=0
source /opt/ros/humble/setup.zsh
source ~/MCLDrone/ovws/install/setup.zsh
source ~/my_px4/install/setup.zsh
nohup ~/start_camera.sh > /tmp/camera_bridge.log 2>&1 & disown
ros2 topic hz /imu0 # 检查IMU话题频率
```

## 2. 启动MCLDrone（在终端 2）
```bash
# 终端 2
ssh ros2@192.168.1200
export ROS_DOMAIN_ID=0
source /opt/ros/humble/setup.zsh
source ~/MCLDrone/ovws/install/setup.zsh
source ~/my_px4/install/setup.zsh
~/MCLDrone/run_live.sh --name flight01 --daemon \ # --name 可以是任何你想要的名字
  --map ~/MCLDrone/maps/z17_5120.png \
  --lat 22.842897 --lon 114.525573 # 起飞点的经纬度
```

## 3. 重新连接后检查(在终端 3中)
```bash
# 终端 3
ssh ros2@192.168.1200
export ROS_DOMAIN_ID=0
source /opt/ros/humble/setup.zsh
source ~/MCLDrone/ovws/install/setup.zsh
source ~/my_px4/install/setup.zsh
~/MCLDrone/run_live.sh --name flight01 --status
# Or
tail -f /tmp/run_live_flight01.console.log
```

## 4. 结束录制（在终端 3中）
```bash
# 终端 3
~/MCLDrone/run_live.sh --name flight01 --stop
```
## 5. 关闭摄像机（在终端 1中）
```bash
# 终端 1
pgrep -af 'cyperstereo|capture_image|mjpeg' # 检查是否有进程在运行
pkill -INT -f 'ros2 launch cyperstereo_ros2_bridge' # 关闭进程
pgrep -af 'cyperstereo|capture_image|mjpeg' # 检查进程是否已关闭（应该为空）
```