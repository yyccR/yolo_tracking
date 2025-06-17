# YOLO + BYTETracker 多目标跟踪

> 本项目集成了 YOLOv8/YOLOv11 目标检测与 BYTETracker 多目标跟踪，基于 ncnn 和 OpenCV，支持 macOS 和 Linux。


![demo](data/demo.gif)


## 目录结构

```
├── common/              # 公共头文件和工具（含 yolo_ncnn.h）
├── yolov8/              # YOLOv8 ncnn 实现与模型
├── yolov11/             # YOLOv11 ncnn 实现与模型
├── byte_tracker/        # BYTETracker 跟踪模块
├── data/                # 测试图片/视频、demo 动图
├── main.cpp             # 主程序，支持命令行切换模型/跟踪器
├── CMakeLists.txt       # 构建配置
└── README.md            # 使用说明
```

## 依赖环境

- OpenCV >= 4.5
- ncnn >= 202306
- CMake >= 3.10
- g++/clang++ (支持 C++17)
- 推荐安装 OpenMP（多线程加速）

### macOS 安装依赖
```bash
brew install opencv ncnn libomp cmake
```

### Linux (Ubuntu) 安装依赖
```bash
sudo apt update
sudo apt install build-essential cmake libopencv-dev libomp-dev
git clone https://github.com/Tencent/ncnn.git
cd ncnn && mkdir build && cd build
cmake .. && make -j$(nproc) && sudo make install
```

> 如需自定义 ncnn 路径，请在 CMakeLists.txt 里修改 NCNN_DIR。

## 编译

```bash
cmake -S . -B build && cmake --build build -j 8
```

- 默认编译 YOLOv8。切换 YOLOv11：
  ```bash
  cmake -S . -B build -DUSE_YOLOV8=OFF && cmake --build build -j 8
  ```

## 模型准备

- YOLOv8/YOLOv11 ncnn 模型导出方法见 yolov8/、yolov11/ 目录下注释。
- 将 .param/.bin 文件放到 yolov8/ 或 yolov11/ 目录。

## 运行参数说明

| 参数         | 说明                       | 默认值                        |
|--------------|----------------------------|-------------------------------|
| --model      | yolov8 或 yolov11          | yolov8                        |
| --param      | NCNN param 文件路径        | yolov8/yolov8s_ncnn.param     |
| --bin        | NCNN bin 文件路径          | yolov8/yolov8s_ncnn.bin       |
| --in         | 输入节点名                 | in0                           |
| --out        | 输出节点名                 | 216 (v8) / 301 (v11)          |
| --prob       | 检测置信度阈值             | 0.25                          |
| --nms        | NMS 阈值                   | 0.45                          |
| --size       | 输入图片缩放尺寸           | 640                           |
| --tracker    | 跟踪算法名（预留扩展）     | BYTETracker                   |
| --video      | 输入视频路径               |                               |
| --image      | 输入图片路径               |                               |

## 运行示例

### 视频跟踪
```bash
# YOLOv8 + BYTETracker
./build/yolo_tracking --model yolov8 --param yolov8/yolov8s_ncnn.param --bin yolov8/yolov8s_ncnn.bin --out 216 --video data/video/track2.mp4

# YOLOv11 + BYTETracker
./build/yolo_tracking --model yolov11 --param yolov11/yolov11n_ncnn.param --bin yolov11/yolov11n_ncnn.bin --out 301 --video data/video/track2.mp4
```

### 图片检测
```bash
./build/yolo_tracking --model yolov8 --param yolov8/yolov8s_ncnn.param --bin yolov8/yolov8s_ncnn.bin --out 216 --image data/xxx.jpg
```

## 常见问题
- **OpenMP/ncnn 报错**：请确认 libomp 已安装，ncnn 路径正确。
- **段错误/崩溃**：请检查输入文件路径、模型文件、OpenCV/ncnn 版本。
- **窗口无响应**：远程服务器请用 `-DWITH_GUI=OFF` 或保存结果图片。



## 致谢
- [ncnn](https://github.com/Tencent/ncnn)
- [YOLOv8/YOLOv11](https://github.com/ultralytics/ultralytics)
- [BYTETracker](https://github.com/ifzhang/ByteTrack)

如有问题欢迎 issue！ 