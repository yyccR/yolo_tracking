#pragma once
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include "common.h"

namespace yolo_ncnn {
// 加载模型和设置输入输出节点
int load(const std::string& bin_file, const std::string& param_file, const std::string& in_blob, const std::string& out_blob);
// 检测接口，所有参数可配置
void detect(const cv::Mat& bgr, std::vector<common::Object>& objects, float prob_threshold, float nms_threshold, int target_size);
} 