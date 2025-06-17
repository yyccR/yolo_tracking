#include <iostream>
#include <opencv2/opencv.hpp>
#include "yolov8/yolov8_ncnn.cpp"
#include "byte_tracker/include/byte_tracker.h"
#include "common/common.h"
#include <getopt.h>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "common/yolo_ncnn.h"

// 动态 include yolov8/yolov11 的 yolo_ncnn.h
// #ifdef USE_YOLOV8
// #include "yolov8/yolo_ncnn.h"
// #else
// #include "yolov11/yolo_ncnn.h"
// #endif

static const char* coco_labels[] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};

int main(int argc, char** argv) {
    std::string model = "yolov8";
    std::string param_file = "yolov8/yolov8s_ncnn.param";
    std::string bin_file = "yolov8/yolov8s_ncnn.bin";
    std::string in_blob = "in0";
    std::string out_blob = "216";
    float prob_threshold = 0.25f;
    float nms_threshold = 0.45f;
    int target_size = 640;
    std::string tracker_name = "BYTETracker";
    std::string video_path;
    std::string image_path;
    // 解析命令行参数
    int opt;
    const char* optstring = "";
    static struct option long_options[] = {
        {"model", required_argument, 0, 0},
        {"param", required_argument, 0, 0},
        {"bin", required_argument, 0, 0},
        {"in", required_argument, 0, 0},
        {"out", required_argument, 0, 0},
        {"prob", required_argument, 0, 0},
        {"nms", required_argument, 0, 0},
        {"size", required_argument, 0, 0},
        {"tracker", required_argument, 0, 0},
        {"video", required_argument, 0, 0},
        {"image", required_argument, 0, 0},
        {0, 0, 0, 0}
    };
    int longindex = 0;
    while ((opt = getopt_long(argc, argv, optstring, long_options, &longindex)) != -1) {
        if (opt == 0) {
            std::string key = long_options[longindex].name;
            if (key == "model") model = optarg;
            else if (key == "param") param_file = optarg;
            else if (key == "bin") bin_file = optarg;
            else if (key == "in") in_blob = optarg;
            else if (key == "out") out_blob = optarg;
            else if (key == "prob") prob_threshold = std::stof(optarg);
            else if (key == "nms") nms_threshold = std::stof(optarg);
            else if (key == "size") target_size = std::stoi(optarg);
            else if (key == "tracker") tracker_name = optarg;
            else if (key == "video") video_path = optarg;
            else if (key == "image") image_path = optarg;
        }
    }
    std::string window_title = model + " + " + tracker_name;
    // 加载模型
    if (yolo_ncnn::load(bin_file, param_file, in_blob, out_blob) != 0) {
        std::cerr << "Failed to load " << model << " model!" << std::endl;
        return -1;
    }
    byte_tracker::BYTETracker tracker(30, 0.3f, 0.1f, 0.8f, 20);
    if (!image_path.empty()) {
        // 单张图片推理
        cv::Mat frame = cv::imread(image_path);
        if (frame.empty()) {
            std::cerr << "Failed to load image: " << image_path << std::endl;
            return -1;
        }
        std::vector<common::Object> objects;
        yolo_ncnn::detect(frame, objects, prob_threshold, nms_threshold, target_size);
        for (const auto& obj : objects) {
            cv::rectangle(frame, obj.rect, cv::Scalar(0, 255, 0), 2);
        }
        cv::imshow(window_title, frame);
        cv::waitKey(0);
        return 0;
    }
    cv::VideoCapture cap;
    if (!video_path.empty()) {
        cap.open(video_path);
        if (!cap.isOpened()) {
            std::cerr << "Failed to open video file: " << video_path << std::endl;
            return -1;
        }
    } else {
        cap.open(0);
        if (!cap.isOpened()) {
            std::cerr << "Failed to open camera!" << std::endl;
            return -1;
        }
    }
    cv::Mat frame;
    while (cap.read(frame)) {
        if (frame.empty()) continue;
        std::vector<common::Object> objects;
        yolo_ncnn::detect(frame, objects, prob_threshold, nms_threshold, target_size);
        std::vector<std::vector<float>> detections;
        std::vector<float> scores;
        std::vector<int> classes;
        for (const auto& obj : objects) {
            detections.push_back({obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height});
            scores.push_back(obj.prob);
            classes.push_back(obj.label);
        }
        auto track_results = tracker.update(detections, scores, classes, frame.cols, frame.rows);

        // ======= 可视化输出 =======
        static std::vector<cv::Scalar> track_colors = {
            cv::Scalar(244, 67, 54), cv::Scalar(233, 30, 99), cv::Scalar(156, 39, 176), cv::Scalar(103, 58, 183),
            cv::Scalar(63, 81, 181), cv::Scalar(33, 150, 243), cv::Scalar(3, 169, 244), cv::Scalar(0, 188, 212),
            cv::Scalar(0, 150, 136), cv::Scalar(76, 175, 80), cv::Scalar(139, 195, 74), cv::Scalar(205, 220, 57),
            cv::Scalar(255, 235, 59), cv::Scalar(255, 193, 7), cv::Scalar(255, 152, 0), cv::Scalar(255, 87, 34),
            cv::Scalar(121, 85, 72), cv::Scalar(158, 158, 158), cv::Scalar(96, 125, 139)
        };
        static std::vector<std::string> coco_labels = {
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"
        };
        int img_w = frame.cols;
        int img_h = frame.rows;
        for (const auto& tr : track_results) {
            float x1 = std::max(0.f, std::min(tr[0], float(img_w - 1)));
            float y1 = std::max(0.f, std::min(tr[1], float(img_h - 1)));
            float x2 = std::max(0.f, std::min(tr[2], float(img_w - 1)));
            float y2 = std::max(0.f, std::min(tr[3], float(img_h - 1)));
            cv::Rect rect(cv::Point(x1, y1), cv::Point(x2, y2));
            int track_id = static_cast<int>(tr[4]);
            float score = tr[5];
            int cls = static_cast<int>(tr[6]);
            cv::Scalar color = track_colors[track_id % track_colors.size()];
            cv::rectangle(frame, rect, color, 2);
            std::string label = std::to_string(track_id) + ": ";
            if (cls >= 0 && cls < coco_labels.size()) label += coco_labels[cls];
            else label += "unknown";
            label += cv::format(" %.2f", score);
            cv::putText(frame, label, cv::Point(rect.x, rect.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
        }
        // 画轨迹
        for (const auto& track_ptr : tracker.get_tracked_stracks()) {
            if (!track_ptr->is_activated()) continue;
            const auto& traj = track_ptr->get_trajectory();
            if (traj.size() > 1) {
                std::vector<cv::Point> pts;
                for (const auto& pt : traj) pts.emplace_back(pt[0], pt[1]);
                cv::polylines(frame, pts, false, track_colors[track_ptr->get_track_id() % track_colors.size()], 2);
            }
        }
        cv::imshow(window_title, frame);
        char key = (char)cv::waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q') break;
    }
    return 0;
} 