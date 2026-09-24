# 配置环境

```bash
git clone https://github.com/Mastopke304/MCLDrone.git
cd MCLDrone

# Conda 训练环境 sivl
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

# 训练

训练是 **x86\_64 开发机** 上的工作流，永远不在无人机上运行：无人机只加载训练产出的微调 checkpoint。

## 本目录中的训练相关文件

| 文件                                            | 作用                                               |
| --------------------------------------------- | ------------------------------------------------ |
| `train_similarity.py`                         | 建库（`--build-bank`）、预训练、建真实航片对（`--build-real`）、微调 |
| `eval_real_patches.py`                        | 真实航片对打分 / 胜率评估                                   |
| `train_config.yml`                            | 全部路径与超参数                                         |
| `sivl/models/orthosimilarity.py`              | BranchNet / DecisionNet 网络定义                     |
| `sivl/utils/utils.py`                         | 地图切片 / tensor 辅助函数                               |
| `bag_reader.py`、`mcl.py`、`orthoprojection.py` | 与运行时共用（真实航片对提取）                                  |
| `environment.yml`                        | 可复现的 conda 环境（`sivl`，py3.9，torch 2.8 cu128）      |

数据和权重 **不进 git**：训练集、zarr patch bank、飞行 bag/ULog、`.pt` checkpoint 都需在开发机上另行准备，路径见 `train_config.yml`。

所有训练/评估命令都在 `sivl` conda 环境中，从 `~/MCLDrone` 目录运行：

```bash
conda activate sivl
cd ~/MCLDrone
```

## 0. 准备数据

```bash
mkdir data && cd data
```

通过谷歌地图API或者谷歌地球Pro（Google Earth Pro）下载用于训练的卫星图片。

打开谷歌地球，找到感兴趣的区块，将视角高度固定在4km到5km之间，重置视角和罗盘（View -> Reset -> Tilt and Compass）以得到向下正视图。在下载地图之前将所有的路网，地名，标签等显示全部取消勾选（Layers栏），否则这些也会一并被下载到图片中。下载选项里，同样取消勾选所有的`Elements`（Map Options里），`Scaling`保持默认的100%，`Styling`保持默认的`Full color base map`。下载图像的分辨率与软件窗口的形状有关，这里推荐将窗口拉为正方形，然后下载分辨率改为`Maximum (8192x8192)`。

启用历史影像，拖动滑块可以选择不同日期的历史影像，因为卫星图像是由多个小区块拼接而成的，因此有些区块的历史图像可能并不是来自同一时期，在地图上表现为有拼接痕迹，尽量选择视野内整个地图都是同一时期拍摄的图像，即没有拼接痕迹。

在一个区块的所有历史影像下载完成之前不要移动视角，否则会污染数据！将一个区块的所有历史图像保存在同一个文件夹内，以时间命名，例如：`area1/2005_8_1.jpg`，其中`2005`为年份，`8`为月份，`1`为这个月的第几张图片，如果同一个月内有多张图片，则增加即可。

最终得到的目录结构应与如下结构大致类似：

```
data
|- area1
   |- 2005_8_1.jpg
   |- 2005_8_2.jpg
   |- 2010_4_1.jpg
   |- ...
|- area2
|- area3
|- ...
|- areaN
```

## 1. 模型训练

配置文件：`train_config.yml`（所有路径与超参数）

```yaml
device: "cuda:0" # CUDA 设备.

data:
  path:
    # ---- 地区数据 ----
    # 根据你的数据目录实际来修改
    trainingdata_bitmaps: # 训练集
      woodbridge: "./data/woodbridge"
      fountainhead: "./data/fountainhead"
      area_1: "./data/area_1"
      area_2: "./data/area_2"
      area_4: "./data/area_4"
      area_5: "./data/area_5"
      area_6: "./data/area_6"
      area_8: "./data/area_8"
      area_9: "./data/area_9"
      area_10: "./data/area_10"
      area_11: "./data/area_11"
      area_12: "./data/area_12"
      area_13: "./data/area_13"
      area_14: "./data/area_14"
      area_15: "./data/area_15"
    testingdata_bitmaps: # 验证集
      area_3: "./data/area_3"
      area_7: "./data/area_7"
    patch_bank: "./data/patch_bank" # 地图区块库保存位置

    map: # 用作大地图的地图文件（这里使用已经拼接好的地图，后续可以使用MapDB动态拼接）
      path: "./z17_5120.png"
      gsd: 1.10 # 大地图的分辨率（米/像素），会影响缩放，一定要提前计算好！
      geo_offset: [-27449088.0, -14586625.0] # 经纬度

    flights: # 自己录制的飞行数据包，用作后续的finetuning，需要包（.db3）与飞行日志（.ulg）
      - name: "bag_0001_20260831_184004"
        run_dir: "./sample_images/mcl_test/bag_0001_20260831_184004_debug"
        ulog: "/your/path/to/bag/bag_0001_20260831_184004/08_13_15.ulg" # 飞行日志用于提供GPS路径真值
        db3: "/home/one/workspace/datasets/bag_0001_20260831_184004/bag_0001_20260831_184004_0.db3" # rosbag db3 (摄像头图像+IMU，用于时间戳对齐)
        alt_min_m: 30.0 # 设置为多少就从多少米的高度开始运行MCL（巡航高度在120米左右）
      - name: "bag_0001_20260908_174711"
        run_dir: "./sample_images/mcl_test/bag_0001_20260831_184004_debug"
        ulog: "/your/path/to/bag/datasets/bag_0001_20260908_174711/log_18_2026-9-8-17-54-50.ulg"
        db3: "/your/path/to/bag/bag_0001_20260908_174711/bag_0001_20260908_174711_0.db3"
        alt_min_m: 30.0
      - name: "bag_0001_20260831_190020"
        run_dir: "./sample_images/mcl_test/bag_0001_20260831_190020_debug"
        ulog: "/your/path/to/bag/bag_0001_20260831_190020/08_32_59.ulg"
        db3: "/your/path/to/bag/bag_0001_20260831_190020/bag_0001_20260831_190020_0.db3"
        alt_min_m: 30.0
  
  # 用于标志相机图像和GPS真值下的大地图图像的模型，一定要是可靠的模型，可以使用预训练好的模型
    label_model_checkpoint: "./checkpoints/your_model.pt" 

training: # 预训练
  checkpoint_saving_path: "./checkpoints/" # 模型权重保存路径
  initial_model_checkpoint: "./checkpoints/your_initial_model.pt" # 起始模型权重，用于断点重联，如果找不到则会从头开始训练
  use_only_network_params: False # False，加载模型权重加上loss和epoch的历史，用于断点重联；
                   # True，只加载模型权重。
  numEpochs: 1000 # 最高训练多少个epoch
  save_every: 100 # 每训练多少个epoch就保存一次权重
  batchsize: 800 # 批次大小，越大训练速度越快但需要越大的显存
  lr: 0.00001 # 学习率，决定学习的速度，非必要不动
  num_workers: 6 # 线程数，并非越高越好，非必要不动
  experiment_name: "train_1" # 此次训练的名字，所有权重都会被保存为 {experiment_name}_epoch_{现在的epoch}_of_{numEpochs}.pt
  
  # 如果之前训练到Epoch 300后中断了，权重保存为了 train_1_epoch_300_of_1000.pt，将这个中断的权重设置为起始权重后 (initial_model_checkpoint=train_1_epoch_300_of_1000.pt)，且use_only_network_params=False，这会加载loss和epoch的历史，因此训练会从300/1000处开始。如果use_only_network_params=True，则之加载权重，训练会从0/1000开始，因此加上之前已经训练好的300个epoch，总共会有1300个epoch。
  
  # 地图区块库的构建参数（非必要不动）
  marginBetweenSamples_px: 0 # grid step = 96 + this (0 = adjacent samples, no stride)
  apShiftingStd: 20 # centerpoint jitter, drawn once per node at --build-bank
  rotationErrorStd_deg: 0.5 # positive-pair rotation noise
  translationErrorStd_px: 0.5 # positive-pair translation noise
  homographyCornerErrorStd_px: 1.0 # positive-pair corner noise
  scaleStd: 0.1 # shared anchor/positive scale draw (clamped <= 1: bank crops are 136 px)
  # --- real-pair negatives in --finetune (cropped from the deployment map) ---
  nearNegativeMin_px: 160 # nearby-negative ring (map px; observed
  nearNegativeMax_px: 600 #  false-peak distances)
  negativeMixture: {near: 0.3, far: 0.5, rotated: 0.2}

finetune: # finetuning，使用真实录制的包，在预训练结束后做
  checkpoint_saving_path: "./checkpoints/"
  initial_model_checkpoint: "./checkpoints/model_pretrained.pt" # 预训练结束后得到的最终权重
  use_only_network_params: True
  numEpochs: 40
  save_every: 10
  batchsize: 800
  lr: 0.00001
  num_workers: 6
  # 下面参数非必要不动
  p_real: 0.3 # fraction of real-pair items per batch
  rotationErrorStd_deg: 3.0 # real-pair positive rotation jitter
  positiveScaleMin: 0.75 # real-pair positive scale jitter (covers coverage
  positiveScaleMax: 1.1 #  estimation error)
  # Cross-flight validation: ALL frames of the listed flights go to realval
  # (the model never trains on them). In-flight frame splits (every 5th frame)
  # memorize the training flight's lighting/terrain and reported "win 100%"
  # while a different flight false-locked (2026-09-09: ft trained on 184004
  # scored truth 1.000 on 184004 but 0.003 on held-out 190020). Empty list =
  # legacy every-5th-frame in-flight split.
  
  # 作为验证集的包
  val_flights: ["bag_0001_20260831_190020"]
  experiment_name: "train_ft_1"

evaluation:
  model_checkpoint: "./checkpoints/train_ft_1_epoch_040_of_40.pt" # finetuning结束之后用于验证的权重

sampledimensions:
  dimension_px: 96 # 模型输入大小，同时也是地图区块库的构建大小，非必要不动
```

### 1.1 构建地图区块库

```bash
# 训练脚本：train_similarity.py
python train_similarity.py --build-bank
```

会按照之前`train_config.yml`文件里的`patch_bank`所指定的路径保存。

需要的硬盘空间和你的数据的数量和大小有关，例如193张图片大约需要45GB的空间。

### 1.2 正式开始预训练

```bash
python train_similarity.py
```

### 1.3 构建真实区块库

在构建前，需要在用于预训练的包上运行一边MCL，得到所有帧的图像与GPS真值，将 ulog 与 bag 时钟对齐（陀螺 z 轴 / 偏航角速度互相关），通过 `geo_offset` 将 GPS 真值映射到地图像素，用标签模型（`train_config.label_model_checkpoint`）为每个调试块生成旋转标签。得分低于 0.5 的帧被丢弃。

推荐只保存飞机巡航阶段的帧，如何运行MCL请看第2节“运行MCL”。在上面的`train_config.yml`文件里我们选定了3个包：

- bag\_0001\_20260831\_184004
- bag\_0001\_20260908\_174711
- bag\_0001\_20260831\_190020

这三个包都要分别运行MCL来保存帧。

保存好后运行：

```bash
python train_similarity.py --build-real
```

每个飞行条目的要求：

- `run_dir`：包含 `replay_log.npz` + `debug/k*_4_patch.png` 的 MCL
  运行输出（即带 `save_debug:=true` 的 MCL 运行）
- `ulog`：含 GPS 真值的飞行日志（`.ulg`）
- `db3`：含 `/imu0` 的 rosbag db3（用于时钟对齐）

输出：`patch_bank/real_pairs.npz`。

### 1.4 真实 + 仿真混合微调

```bash
python train_similarity.py --finetune
```

从 `train_config.finetune.initial_model_checkpoint` 初始化。实验名 `train_ft_1` →`checkpoints/train_ft_1_epoch_040_of_40.pt`（部署用检查点；也是新的 `mcl.py`/launch 默认值）。混合比例：`finetune.p_real`的真实数据对，其余为仿真块库数据对。每个 epoch 跟踪两个验证集：

- `simval` — 必须保持 acc 1.000
- `realval` — 跨飞行验证：`train_config.finetune.val_flights` 中列出的飞行
  （当前为 190020）的**全部**帧；模型从不在这上面训练。这是部署
  门禁 — 期望留出集 pos \~0.9+。`train_config.data.path.flights` 中的所有飞行都通过
  `--build-real` 生成数据对；`train_config.finetune.val_flights` 只控制哪些被排除在训练外。
  最终验收：在留出飞行上跑完整 MCL bag 回放（比离线块指标更强），
  外加一条从未用过的第 3 条飞行为真正的测试集。

### 1.5 冒烟测试

```bash
python train_similarity.py --epochs 2 --max-batches 30
python train_similarity.py --finetune --epochs 2 --max-batches 30
```

### 1.6 在真实块上评估（真值 vs 假峰）

```bash
# 完整评估：GPS 真值处 + 近/远假位置处的 2 度旋转扫描
python eval_real_patches.py --frames 40 --save-sb 6 \
  --ckpt checkpoints/train_ft_1_epoch_040_of_40.pt

# 失败诊断：尺度扫描 + 块统计 + 嵌入域差距
python eval_real_patches.py --profile --frames 40 --ckpt checkpoints/train_ft_1_epoch_040_of_40.pt
```

## 2. 运行 MCL

ROS2 Jazzy 环境：

```bash
source /opt/ros/jazzy/setup.bash
source ~/workspace/ovws/install/setup.bash
```

**重要**：重新启动前先杀死残留节点（同 `out_dir` 的孤儿节点会损坏
`replay_log.npz`）：

```bash
pkill -f mcl_node
```

### 2.1 标准运行（VIO + MCL + bag 回放）

```bash
ros2 launch ov_mcl mcl_localization.launch.py bag:=<bag_dir>
```

示例：

```bash
ros2 launch ov_mcl mcl_localization.launch.py \
  bag:=/home/one/workspace/datasets/bag_0001_20260831_184004
```

默认值：`vio_config=cyperstereo_012_752x480_equi`，地图
`z17_5120.png`，检查点
`checkpoints_huizhou/huizhou_ft_1_epoch_040_of_40.pt`。

### 常用 launch 参数

| 参数                            | 默认值                            | 含义                                                                                                                                                                                                                        |
| ----------------------------- | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `bag:=`                       | —                              | 要回放的 rosbag2 目录（回放测试）。同时在 `replay.png` 上启用 GPS 真值叠加：bag 旁的 ULog（`*.ulg`）通过 /imu0 陀螺 z 轴 ↔ 偏航角速度互相关对齐到 bag 时钟，真值轨迹（黄色虚线）绘制在轨迹面板上，标题中给出中位/p90 误差。滤波器**从不**使用 GPS（仅用于绘图）。bag 旁无 `.ulg`/`.db3` 或对齐较弱（corr < 0.3）→ 不带真值绘图并给出警告 |
| `vio_config:=`                | `cyperstereo_012_752x480_equi` | VIO 配置名或完整路径                                                                                                                                                                                                              |
| `checkpoint:=`                | train\_ft\_1                   | 相似度模型检查点                                                                                                                                                                                                                  |
| `map_path:=`                  | `z17_5120.png`                 | 部署地图                                                                                                                                                                                                                      |
| `out_dir:=`                   | `.../mcl_test/online`          | 输出目录（replay\_log.npz、绘图）                                                                                                                                                                                                  |
| `stride:=`                    | `5`                            | 每隔 N 个相机帧处理一次                                                                                                                                                                                                             |
| `frame_start:=`               | `0`                            | 首个处理的相机帧（跳过起飞；0 = 从头开始）                                                                                                                                                                                                   |
| `frame_end:=`                 | `0`                            | 最后处理的相机帧 + 1（跳过降落；0 = 到结尾）                                                                                                                                                                                                |
| `n_particles:=`               | `1000`                         | 蒙特卡洛粒子数                                                                                                                                                                                                                   |
| `init_mode:=`                 | `scan`                         | `scan` / `point` / `click`                                                                                                                                                                                                |
| `init_lat:=` / `init_lon:=`   | `-999.0`                       | `point` 模式的 WGS-84 初始纬/经度（度）— **覆盖** `init_px_x/y`                                                                                                                                                                        |
| `init_px_x:=` / `init_px_y:=` | `0`                            | `point` 模式的初始点（地图像素）                                                                                                                                                                                                      |
| `map_geo_offset_x/y:=`        | z17\_5120.png 偏移               | 地理配准（用于经纬度换算）：局部像素 = 全局 z17 墨卡托像素 + 偏移（对由 z17 切片拼接、左上角切片原点为 `(tx,ty)` 的地图：`(-tx*256, -ty*256)`）                                                                                                                           |
| `alt_anchor:=`                | `0.0`                          | 加到 VIO z 上的高度偏移                                                                                                                                                                                                           |
| `save_debug:=`                | `false`                        | 保存逐帧流水线图像（`--build-real` 需要）                                                                                                                                                                                              |
| `debug_stride:=`              | `1`                            | 每隔 N 个已处理帧保存一次调试图像                                                                                                                                                                                                        |

### 2.2 分开运行 VIO、RVIZ 与 MCL

用于独立检查各阶段（例如在启动 MCL 前用 RVIZ 查看 VIO 轨迹）。
每个终端单独开，均需 source ROS2 环境：

```bash
source /opt/ros/jazzy/setup.bash && source ~/workspace/ovws/install/setup.bash
```

如遇 `PermissionError: [Errno 13] Permission denied: '/home/one/.ros/log/...'`，
先重定向日志目录：

```bash
export ROS_LOG_DIR=/tmp/ros_log
```

**终端 1 — bag 回放**（实时相机可跳过）：

```bash
ros2 bag play /home/one/workspace/datasets/bag_0001_20260831_184004
```

**终端 2 — 仅 VIO**（SchurVINS/OpenVINS，订阅`/cam0/image_raw`、`/cam1/image_raw`、`/imu0`；发布`/ov_msckf/odomimu`）：

```bash
ros2 launch ov_msckf subscribe.launch.py config:=cyperstereo_012_752x480_equi rviz_enable:=true
```

加 `rviz_enable:=true` 可同时启动带 VIO 显示配置的 RVIZ，或单独运行RVIZ（见下）以同时观察 VIO 与 MCL。

**终端 3 — 仅 MCL**（订阅 `/cam0/image_raw` +
`/ov_msckf/odomimu`；VIO 开始发布后再启动）：

```bash
ros2 run ov_mcl mcl_node --ros-args \
  -p map_path:=./z17_5120.png \
  -p checkpoint:=./checkpoints_huizhou/huizhou_ft_1_epoch_040_of_40.pt \
  -p calib_path:=/home/one/workspace/ovws/install/ov_msckf/share/ov_msckf/config/cyperstereo_012_752x480_equi/kalibr_imucam_chain.yaml \
  -p out_dir:=./sample_images/mcl_test/online_separate \
  -p stride:=5 \
  -p n_particles:=1000 \
  -p init_mode:=point \
  -p init_lat:=22.842866 \
  -p init_lon:=114.525564
```

其他有用的 `--ros-args -p` 参数（含义同上表）：`init_px_x`、`init_px_y`
（配合 `init_mode:=point`）、`alt_anchor`、`save_debug:=true`、
`debug_stride`。

参考初始点：

- Bag 184004：`init_lat:=22.842866 init_lon:=114.525564`
- Bag 19002 — MCL 节点需要VIO 里程计驱动其预测步0：`init_lat:=22.842897 init_lon:=114.525573`
- Bag 174711：`init_lat:=22.842871 init_lon:=114.525563`

注意：

- 启动顺序很重要：先 bag/VIO，等 `/ov_msckf/odomimu` 有消息
  （`ros2 topic hz /ov_msckf/odomimu`）再启动 MCL。
- 运行时录制 VIO 里程计（供离线分析）：
  `ros2 bag record /ov_msckf/odomimu -o vio_odom`

### 2.3 真值锚定 + 保存调试图像的运行（bag\_0001\_20260831\_184004，相机帧 2000-8200，用于构建真实数据对）

调试文件名中的 `k` 就是相机帧索引（k02000 = bag 相机帧 2000；
stride 只筛选哪些帧出现，不重新编号）。`sivl`检查点只是让
MCL 跑起来 — `--build-real` 的训练块/真值来自 bag + ulog，而非该模型。

```bash
# 按地图像素：
ros2 launch ov_mcl mcl_localization.launch.py \
  bag:=<bag_dir> \
  init_mode:=point init_px_x:=2686 init_px_y:=2728 \
  save_debug:=true \
  out_dir:=./sample_images/mcl_test/my_run \
  frame_start:=2000 frame_end:=8201

# 或按 WGS-84 经纬度（覆盖 init_px；与上等效）：
ros2 launch ov_mcl mcl_localization.launch.py \
  bag:=<bag_dir> \
  init_mode:=point init_lat:=22.842871 init_lon:=114.525563 \
  save_debug:=true \
  out_dir:=./sample_images/mcl_test/my_run \
  frame_start:=2000 frame_end:=8201
```

- `frame_end` 为**开区间**（`frame_idx < frame_end`）：8201 包含 k08200
- stride=5 时：处理帧 k02000、k02005、…、k08200（1241 帧，全部高于
  30 m 高度门限 — 已通过 ulog 验证）
- 初始点 = 帧时刻 2000 处的无人机 GPS 位置（ulog：纬度 22.842866、
  经度 114.525564 → 地图像素 (2684,2728)；此时无人机仍在爬升的
  近垂直状态，高度约 58 m）
- VIO 仍从 bag 开头消费所有帧（只有 MCL 跳过前 2000 帧），因此
  首个处理帧之前 VIO 已充分预热
- Bag 184004：frame\_start:=2000 frame\_end:=8201
- Bag 190020：frame\_start:=1100 frame\_end:=14801
- Bag 174711：frame\_start:=900 frame\_end:=10091

然后将该运行加入 `train_config.yml` 的 `data.path.flights`，
再重新运行阶段 1.3/1.4。

***

## 3. 重建工作空间（代码修改后）

工作空间：`/home/one/workspace/ovws`（包：`ov_mcl`、`ov_SchurVINS`）。

修改 `src/ov_mcl/` 或 `src/ov_SchurVINS/` 中的 Python/launch/配置
文件后，重建以使 `install/` 空间生效：

```bash
source /opt/ros/jazzy/setup.bash
cd /home/one/workspace/ovws
colcon build --packages-select ov_mcl --symlink-install
```

- `--symlink-install`：安装符号链接而非拷贝 — Python 源码修改
  （`ov_mcl/ov_mcl/*.py`）无需重建即生效；只有 launch 文件 / 入口点 /
  非 Python 文件需要重建
- 全量重建：`colcon build --symlink-install`
- 重建 SchurVINS（C++，VIO 源码修改后 — 较慢）：
  ```bash
  cd /home/one/workspace/ovws
  colcon build --packages-select ov_msckf
  ```

任何重建后重新 source：

```bash
source ~/workspace/ovws/install/setup.bash
```

注意：`mcl_node` 直接从 `.` 导入
`mcl.py`、`bag_mcl.py` 等（见
[mcl\_node.py](/home/one/workspace/ovws/src/ov_mcl/ov_mcl/mcl_node.py)
中的 `GNSS_DIR`）— 修改这些文件无需重建，重启节点即可。

***

## 4. 典型完整流程（从零开始 / 新区域）

当前区域的区域导出、飞行数据与配置均已就绪 — 从第 4 步开始：

1. 准备多时相区域导出：Google-Earth 固定视口（8192×8192，`data/areaN/` 下每区域一个文件夹，内含 `year_month_N.jpg` 图层，每区域 ≥2 层）。训练与验证区域必须空间不相交。通过将一层与部署地图互相关测量 `bitmap_gsd`（惠州：0.60）
2. 录制一次飞行 bag + ulog；带 `save_debug:=true` 跑一次 MCL（见上文"真值锚定运行"）
3. 编辑 `train_config.yml`：区域文件夹、`bitmap_gsd`、部署地图路径、`gsd`、`geo_offset`、飞行数据
4. `python train_similarity.py --build-bank`（重建块库）
5. `python train_similarity.py` — 预训练 → `huizhou_mt_1_epoch_XXXXX_of_1000.pt`
6. `python train_similarity.py --build-real` — 重建 `real_pairs.npz`（与预训练无关；只需标签模型 + 飞行调试运行）
7. `python train_similarity.py --finetune` — 从预训练输出初始化（配置已指向它）→ `huizhou_ft_mt_1_epoch_040_of_40.pt`
8. `python eval_real_patches.py --ckpt checkpoints_huizhou/huizhou_ft_mt_1_epoch_040_of_40.pt`— 期望 win \~100%，留出飞行真值得分高
9. 部署：`ros2 launch ov_mcl mcl_localization.launch.py bag:=<bag>`（默认检查点为 mt 微调输出）

***

## 备注

- 由 z17 切片拼接、左上角切片原点为 `(tx, ty)` 的地图的 `geo_offset`：`[-tx*256, -ty*256]`
- 评估用的旋转扫描必须用 2 度步长（模型对旋转敏感；10 度步长会完全漏掉峰值）
- GPU 训练：长时间运行可按需 `nohup ... &` 启动；每个检查点约 212 MB
- 检查点位于 `checkpoints_huizhou/`；默认部署检查点设置在[`mcl.py`](mcl.py) 的 `CHECKPOINT` 与 launch 文件中

# 在ROS2 Bag中运行MCLDrone

## 1. 运行ov\_SchurVINS + MCL -> Odometry

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
NAME=flight01 # 你可以选择任何你想要的名字
mkdir -p ~/ros2bag/mcl_runs ~/ros2bag/${NAME}_flightlog
nohup ~/start_camera.sh > ~/ros2bag/${NAME}_flightlog/camera_bridge.log 2>&1 & disown
ros2 topic hz /imu0 # 检查IMU话题频率
tail -f ~/ros2bag/${NAME}_flightlog/camera_bridge.log
```

## 2. 启动MCLDrone

可选启动方式：

- 使用脚本启动MCLDrone（跳转2-1）
- 手动启动MCLDrone（跳转2-2）

### 2-1. 启动MCLDrone（脚本）

#### 2-1-1. 使用脚本启动MCLDrone
选择其中一个：
 - 仅观察MCLDrone的Odometry
 - 将MCLDrone的Odometry推送给EV，在QGC中设定好计划，保存为`*.plan`

仅观察MCLDrone的Odometry：

```bash
~/MCLDrone/run_live.sh --name flight01 --daemon \ # --name 可以是任何你想要的名字
  --map ~/MCLDrone/maps/z17_5120.png \
  --lat 22.842897 --lon 114.525573 # 起飞点的经纬度
```

将MCLDrone的Odometry推送给EV，在QGC中设定好计划，保存为`*.plan`：

```bash
~/MCLDrone/run_live.sh --name gpstest01 --daemon \
  --map ~/MCLDrone/maps/z17_5120.png --lat 22.842897 --lon 114.525573 \
  --manager --qgc-plan path/to/*.plan --cruise-alt 50 --v-max 10
```

#### 2-1-2. 重新连接后检查

```bash
~/MCLDrone/run_live.sh --name flight01 --status
# Or
tail -f ~/ros2bag/flight01_flightlog/console.log
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

### 2-2. 启动MCLDrone（手动分别启动）

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

选择其中一个：
 - 仅观察MCLDrone的Odometry
 - 将MCLDrone的Odometry推送给EV，在QGC中设定好计划，保存为`*.plan`

仅观察MCLDrone的Odometry：

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

将MCLDrone的Odometry推送给EV，在QGC中设定好计划，保存为`*.plan`：

```bash
nohup python3 ~/MCLDrone/mission_manager.py --ros-args \
  -p mission_mode:=waypoints \
  -p ev_frame:=map \
  -p map_geo_offset_x:=-27449088.0 -p map_geo_offset_y:=-14586624.0 \
  -p map_zoom:=17 -p map_gsd:=1.1 -p map_center_px:=2560.0 \
  -p map_center_lat:=22.8445297 -p map_center_lon:=114.5242310 \
  -p cruise_alt:=120.0 -p v_max:=10.0 \
  -p 'qgc_plan:=/home/ros2/missions/my_trace.plan' \
  > ~/ros2bag/${NAME}_flightlog/manager.log 2>&1 &
disown

#监控与停止：
tail -f ~/ros2bag/${NAME}_flightlog/manager.log
ROS_DOMAIN_ID=0 ros2 topic echo /mission/state
pkill -INT -f mission_manager.py     # 先停它，再停 MCL/recorder
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
pkill -INT -f '/lib/ov_mcl/mcl_node'
pkill -INT -f 'ros2 launch ov_mcl'
pkill -INT -f 'ros2 bag record'
pkill -f run_subscribe_msckf
pkill -INT -f fcu_pose_bridge.py
pkill -INT -f MicroXRCEAgent
ss -ulnp | grep 8888 # 需要为空
```

## run\_live.sh 参数表

### 通用 / 会话控制

| 参数                | 默认值          | 说明                                                       |
| ----------------- | ------------ | -------------------------------------------------------- |
| `--name <s>`      | `live_<时间戳>` | 运行名；bag、MCL 输出、PID、日志均按它命名                               |
| `--domain <n>`    | `0`          | `ROS_DOMAIN_ID`；FCU、相机、recorder 必须同域                     |
| `--daemon` / `-D` | 关            | 新会话脱离 SSH（断连继续运行）；日志到 `$BAG_ROOT/<name>_flightlog/console.log` |
| `--stop`          | —            | 停止同名运行（manager→MCL npz→recorder→其它）                      |
| `--status`        | —            | 查看进程与日志尾部                                                |
| `-h` / `--help`   | —            | 帮助                                                       |

### 输出路径

| 参数                    | 默认值                            | 说明                                      |
| --------------------- | ------------------------------ | --------------------------------------- |
| `--bag-root <dir>`    | `~/ros2bag`                    | bag 根目录，输出 `<name>_img` 与 `<name>_data` |
| `--mcl-root <dir>`    | `~/ros2bag/mcl_runs`           | MCL 输出目录（replay\_log.npz / replay.png）  |
| `--map <file>`        | `~/MCLDrone/maps/z17_5120.png` | 部署地图 PNG                                |
| `--off-x <f>`         | `-27449088.0`                  | 地图 web-mercator geo offset X            |
| `--off-y <f>`         | `-14586624.0`                  | 地图 web-mercator geo offset Y            |
| `--checkpoint <file>` | 空                              | 自定义 `.pt`；空则用 launch 默认微调模型             |

### MCL 初始化 / 处理

| 参数                    | 默认值                        | 说明                                                       |
| --------------------- | -------------------------- | -------------------------------------------------------- |
| `--init-mode <s>`     | `point`                    | `point`（经纬度点初始化）或 `scan`                                 |
| `--lat <f>`           | `-999`（point 必填）           | 起飞点纬度 WGS-84                                             |
| `--lon <f>`           | `-999`（point 必填）           | 起飞点经度 WGS-84                                             |
| `--radius <f>`        | `300.0`                    | 初始点搜索半径 m                                                |
| `--stride <n>`        | `5`                        | 相机帧处理步长                                                  |
| `--yaw-cal <a,b>`     | `-1,1.3`                   | 罗盘 NED 航向→地图偏航 `yaw_mcl = a*hdg + b`                     |
| `--fcu-topic <topic>` | `/fcu/local_position/pose` | MCL 订阅的 FCU ENU PoseStamped 话题                           |
| `--extra <s>`         | 空                          | 原样追加的 launch 参数，如 `"n_particles:=1500 save_debug:=true"` |

### 录制

默认启动**两个** recorder：图像走 SHM、高频小消息走 UDP-only
（不拆分会把约 200 Hz 的 /imu0 录成 \~25 Hz）。

| 参数                  | 默认值                                                 | 说明                                 |
| ------------------- | --------------------------------------------------- | ---------------------------------- |
| `--img-topics <s>`  | `/cam0/image_raw /cam1/image_raw`                   | 图像 bag（SHM）话题，空格分隔                 |
| `--data-topics <s>` | imu0 / odomimu / mcl odom+health / fcu pose / fmu×4 | 数据 bag（UDP-only）话题，空格分隔            |
| `--topics <s>`      | 空                                                   | 非空时只开**一个** recorder 录这些话题（覆盖双路默认） |
| `--no-record`       | —                                                   | 不启动任何 recorder                     |

默认数据话题全集：
`/imu0 /ov_msckf/odomimu /mcl/odom /mcl/health /fcu/local_position/pose
/fmu/out/vehicle_local_position_v1 /fmu/out/vehicle_attitude
/fmu/out/vehicle_status_v1 /fmu/in/vehicle_visual_odometry`

### 组件开关

| 参数            | 默认          | 说明                                  |
| ------------- | ----------- | ----------------------------------- |
| `--no-agent`  | 默认启动 agent  | 不自动启动 MicroXRCE agent（已手动起时用）       |
| `--no-bridge` | 默认启动 bridge | 不起 fcu\_pose\_bridge（MAVROS 机型或直连时） |

已在运行时会自动跳过重复启动 agent（检测到 `/fmu/out/vehicle_attitude`）。

### mission\_manager（OFFBOARD 自动驾驶，默认关）

>  加 `--manager` 后，节点在 `/mcl/odom` 有效且 guard 进入 TAKEOFF 时
> 会**自行解锁并切换 OFFBOARD**。测试时保持 RC/QGC 可随时接管；停止先停
> manager（`--stop` 已按此顺序处理）。

| 参数                           | 默认值           | 说明                                                   |
| ---------------------------- | ------------- | ---------------------------------------------------- |
| `--manager` / `--no-manager` | 关             | 启动 / 不启动 mission\_manager                            |
| `--mgr-mode <s>`             | `waypoints`   | `waypoints`（按航点）或 `feedforward`（跟随 EV 速度）            |
| `--qgc-plan <file>`          | 空             | QGC `.plan` 文件；home 自动作为 local origin                |
| `--waypoints-ll <s>`         | 空             | 经纬度航点 `lat,lon[,alt,speed]; ...`                     |
| `--waypoints <s>`            | 空             | map-ENU 米航点 `x,y; x,y`（旧格式，统一巡航高度/速度）                |
| `--origin-x <f>`             | `0`           | 起飞点 map 东向米坐标（QGC plan 时被 plan 的 home 覆盖）            |
| `--origin-y <f>`             | `0`           | 起飞点 map 北向米坐标（同上）                                    |
| `--cruise-alt <f>`           | `50.0`        | 巡航/目标高度 m（无逐点高度时使用）                                  |
| `--v-max <f>`                | `12.0`        | 最大水平速度 m/s                                           |
| `--ev-frame <s>`             | `map`         | EV 坐标约定：`map`（减 local\_origin）或 `first_fix`（减首个 fix） |
| `--map-zoom <n>`             | `17`          | 部署地图切片 zoom（经纬度航点换算）                                 |
| `--map-gsd <f>`              | `1.1`         | 部署地图 m/px                                            |
| `--map-center-px <f>`        | `2560.0`      | 地图中心像素（z17\_5120 边长之半）                               |
| `--map-center-lat <f>`       | `22.8445297`  | EKF2 global origin 纬度                                |
| `--map-center-lon <f>`       | `114.5242310` | EKF2 global origin 经度                                |
| `--mgr-extra <s>`            | 空             | 原样追加的 manager `-p key:=value` 参数                     |

航点来源优先级：`--qgc-plan` → `--waypoints-ll` → `--waypoints`。

QGC `.plan` 支持的指令：`NAV_TAKEOFF`（取 home/local origin）、
`NAV_WAYPOINT`（位置/相对或 AMSL 高度）、`NAV_LAND`（忽略，由 guard
统一 AUTO.LAND）、`DO_CHANGE_SPEED`（后续航点速度）。不支持 QGC 的逐点
偏航/接受半径/测绘网格——速度 offboard 控制器沿航迹指向、用内部接受半径。

## 产物与日志

| 内容     | 位置                                                                    |
| ------ | --------------------------------------------------------------------- |
| 图像 bag | `$BAG_ROOT/<name>_img/`                                               |
| 数据 bag | `$BAG_ROOT/<name>_data/`                                              |
| MCL 结果 | `$MCL_ROOT/<name>/`（replay\_log.npz、replay.png）                       |
| 各组件日志  | `$BAG_ROOT/<name>_flightlog/{agent,bridge,record_*,mcl_launch,manager}.log` |
| 守护态控制台 | `$BAG_ROOT/<name>_flightlog/console.log`                                    |
| PID    | `$BAG_ROOT/<name>_flightlog/run.pid`                                            |

## 停止顺序（`--stop` 内部逻辑）

1. SIGINT `mission_manager.py`（停止 OFFBOARD 指令，飞控交还控制）；
2. 直接 SIGINT `mcl_node`（等最多 40 s 刷写 replay\_log.npz）；
3. SIGINT 两个 recorder（flush mcap 尾部，避免无用尾巴数据）；
4. 清理 SchurVINS、bridge、agent 残留；
5. 报告 npz 是否写出（低于约 44 m 高度 0 帧处理属正常，不写 npz）。

