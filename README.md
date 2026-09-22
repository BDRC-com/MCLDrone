# 在ROS2 Bag中运行MCLDrone
## 1. 配置环境
```bash
git clone https://github.com/Mastopke304/MCLDrone.git
cd MCLDrone

# pip
pip install -r requirements.txt
# Conda
conda env create -f environment.yml
```
创建好环境后，需要将如下文件放置在制定地点：
```bash
# 模型权重 huizhou_ft_mt_1_epoch_040_of_40.pt
cd checkpoints_huizhou/
# 地图 z17_5120.png
cd maps/
# ov_SchurVINS
cd ovws/src/
git clone https://github.com/BDRC-com/ov_SchurVINS.git
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

## 3. 可视化Odometry和GPS Truth
```bash
python3 ~/MCLDrone/plot_odom_vs_gps.py \
    $log_out_dir \  # MCL run directory
    --bag $bag_out_dir \  # Odometry records
    --fly ~/datasets/bag_0001_20260831_184004  # GPS truth
```
会在`$log_out_dir`目录下生成可视化结果。

# 在线运行MCLDrone
*WIP*