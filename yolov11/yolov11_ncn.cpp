// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2025 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

// 1. install
//      pip3 install -U ultralytics pnnx ncnn
// 2. export yolo11 torchscript
//      yolo export model=yolo11n.pt format=torchscript
// 3. convert torchscript with static shape
//      pnnx yolo11n.torchscript
// 4. modify yolo11n_pnnx.py for dynamic shape inference
//      A. modify reshape to support dynamic image sizes
//      B. permute tensor before concat and adjust concat axis
//      C. drop post-process part
//      before:
//          v_235 = v_204.view(1, 144, 6400)
//          v_236 = v_219.view(1, 144, 1600)
//          v_237 = v_234.view(1, 144, 400)
//          v_238 = torch.cat((v_235, v_236, v_237), dim=2)
//          ...
//      after:
//          v_235 = v_204.view(1, 144, -1).transpose(1, 2)
//          v_236 = v_219.view(1, 144, -1).transpose(1, 2)
//          v_237 = v_234.view(1, 144, -1).transpose(1, 2)
//          v_238 = torch.cat((v_235, v_236, v_237), dim=1)
//          return v_238
//      D. modify area attention for dynamic shape inference
//      before:
//          v_95 = self.model_10_m_0_attn_qkv_conv(v_94)
//          v_96 = v_95.view(1, 2, 128, 400)
//          v_97, v_98, v_99 = torch.split(tensor=v_96, dim=2, split_size_or_sections=(32,32,64))
//          v_100 = torch.transpose(input=v_97, dim0=-2, dim1=-1)
//          v_101 = torch.matmul(input=v_100, other=v_98)
//          v_102 = (v_101 * 0.176777)
//          v_103 = F.softmax(input=v_102, dim=-1)
//          v_104 = torch.transpose(input=v_103, dim0=-2, dim1=-1)
//          v_105 = torch.matmul(input=v_99, other=v_104)
//          v_106 = v_105.view(1, 128, 20, 20)
//          v_107 = v_99.reshape(1, 128, 20, 20)
//          v_108 = self.model_10_m_0_attn_pe_conv(v_107)
//          v_109 = (v_106 + v_108)
//          v_110 = self.model_10_m_0_attn_proj_conv(v_109)
//      after:
//          v_95 = self.model_10_m_0_attn_qkv_conv(v_94)
//          v_96 = v_95.view(1, 2, 128, -1)
//          v_97, v_98, v_99 = torch.split(tensor=v_96, dim=2, split_size_or_sections=(32,32,64))
//          v_100 = torch.transpose(input=v_97, dim0=-2, dim1=-1)
//          v_101 = torch.matmul(input=v_100, other=v_98)
//          v_102 = (v_101 * 0.176777)
//          v_103 = F.softmax(input=v_102, dim=-1)
//          v_104 = torch.transpose(input=v_103, dim0=-2, dim1=-1)
//          v_105 = torch.matmul(input=v_99, other=v_104)
//          v_106 = v_105.view(1, 128, v_95.size(2), v_95.size(3))
//          v_107 = v_99.reshape(1, 128, v_95.size(2), v_95.size(3))
//          v_108 = self.model_10_m_0_attn_pe_conv(v_107)
//          v_109 = (v_106 + v_108)
//          v_110 = self.model_10_m_0_attn_proj_conv(v_109)
// 5. re-export yolo11 torchscript
//      python3 -c 'import yolo11n_pnnx; yolo11n_pnnx.export_torchscript()'
// 6. convert new torchscript with dynamic shape
//      pnnx yolo11n_pnnx.py.pt inputshape=[1,3,640,640] inputshape2=[1,3,320,320]
// 7. now you get ncnn model files
//      mv yolo11n_pnnx.py.ncnn.param yolo11n.ncnn.param
//      mv yolo11n_pnnx.py.ncnn.bin yolo11n.ncnn.bin

// the out blob would be a 2-dim tensor with w=144 h=8400
//
//        | bbox-reg 16 x 4       | per-class scores(80) |
//        +-----+-----+-----+-----+----------------------+
//        | dx0 | dy0 | dx1 | dy1 |0.1 0.0 0.0 0.5 ......|
//   all /|     |     |     |     |           .          |
//  boxes |  .. |  .. |  .. |  .. |0.0 0.9 0.0 0.0 ......|
//  (8400)|     |     |     |     |           .          |
//       \|     |     |     |     |           .          |
//        +-----+-----+-----+-----+----------------------+
//

#include "../common/yolo_ncnn.h"
#include "net.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <iostream>
#include "../common/common.h"

namespace yolo_ncnn {
namespace {
    ncnn::Net net;
    std::string last_bin_file;
    std::string last_param_file;
    std::string input_blob_name = "in0";
    std::string output_blob_name = "301";
}

int load(const std::string& bin_file, const std::string& param_file, const std::string& in_blob, const std::string& out_blob) {
    if (bin_file == last_bin_file && param_file == last_param_file)
        return 0;
    net.clear();
    net.opt.use_vulkan_compute = true;
    if (net.load_param(param_file.c_str()) != 0) return -1;
    if (net.load_model(bin_file.c_str()) != 0) return -1;
    last_bin_file = bin_file;
    last_param_file = param_file;
    input_blob_name = in_blob;
    output_blob_name = out_blob;
    return 0;
}

static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static void qsort_descent_inplace(std::vector<common::Object>& objects, int left, int right) {
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].prob;
    while (i <= j) {
        while (objects[i].prob > p) i++;
        while (objects[j].prob < p) j--;
        if (i <= j) {
            std::swap(objects[i], objects[j]);
            i++; j--;
        }
    }
    if (left < j) qsort_descent_inplace(objects, left, j);
    if (i < right) qsort_descent_inplace(objects, i, right);
}
static void qsort_descent_inplace(std::vector<common::Object>& objects) {
    if (objects.empty()) return;
    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<common::Object>& objects, std::vector<int>& picked, float nms_threshold, bool agnostic = false) {
    picked.clear();
    const int n = objects.size();
    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) areas[i] = objects[i].rect.area();
    for (int i = 0; i < n; i++) {
        const auto& a = objects[i];
        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++) {
            const auto& b = objects[picked[j]];
            if (!agnostic && a.label != b.label) continue;
            float inter_area = (a.rect & b.rect).area();
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            if (inter_area / union_area > nms_threshold) keep = 0;
        }
        if (keep) picked.push_back(i);
    }
}

static void generate_proposals(const ncnn::Mat& pred, const std::vector<int>& strides, const ncnn::Mat& in_pad, float prob_threshold, std::vector<common::Object>& objects) {
    const int w = in_pad.w;
    const int h = in_pad.h;
    int pred_row_offset = 0;
    for (size_t i = 0; i < strides.size(); i++) {
        const int stride = strides[i];
        const int num_grid_x = w / stride;
        const int num_grid_y = h / stride;
        const int num_grid = num_grid_x * num_grid_y;
        for (int idx = 0; idx < num_grid; idx++) {
            int y = idx / num_grid_x;
            int x = idx % num_grid_x;
            const ncnn::Mat pred_grid = pred.row_range(pred_row_offset + idx, 1);
            int label = -1;
            float score = -FLT_MAX;
            const int reg_max_1 = 16;
            const int num_class = pred.w - reg_max_1 * 4;
            const ncnn::Mat pred_score = pred_grid.range(reg_max_1 * 4, num_class);
            for (int k = 0; k < num_class; k++) {
                float s = pred_score[k];
                if (s > score) { label = k; score = s; }
            }
            score = sigmoid(score);
            if (score >= prob_threshold) {
                ncnn::Mat pred_bbox = pred_grid.range(0, reg_max_1 * 4).reshape(reg_max_1, 4);
                ncnn::Layer* softmax = ncnn::create_layer("Softmax");
                ncnn::ParamDict pd;
                pd.set(0, 1); pd.set(1, 1);
                softmax->load_param(pd);
                ncnn::Option opt; opt.num_threads = 1; opt.use_packing_layout = false;
                softmax->create_pipeline(opt);
                softmax->forward_inplace(pred_bbox, opt);
                softmax->destroy_pipeline(opt);
                delete softmax;
                float pred_ltrb[4];
                for (int k = 0; k < 4; k++) {
                    float dis = 0.f;
                    const float* dis_after_sm = pred_bbox.row(k);
                    for (int l = 0; l < reg_max_1; l++) dis += l * dis_after_sm[l];
                    pred_ltrb[k] = dis * stride;
                }
                float pb_cx = (x + 0.5f) * stride;
                float pb_cy = (y + 0.5f) * stride;
                float x0 = pb_cx - pred_ltrb[0];
                float y0 = pb_cy - pred_ltrb[1];
                float x1 = pb_cx + pred_ltrb[2];
                float y1 = pb_cy + pred_ltrb[3];
                common::Object obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = x1 - x0;
                obj.rect.height = y1 - y0;
                obj.label = label;
                obj.prob = score;
                objects.push_back(obj);
            }
        }
        pred_row_offset += num_grid;
    }
}

void detect(const cv::Mat& bgr, std::vector<common::Object>& objects, float prob_threshold, float nms_threshold, int target_size) {
    int img_w = bgr.cols;
    int img_h = bgr.rows;
    int w = img_w, h = img_h;
    float scale = 1.f;
    if (w > h) { scale = (float)target_size / w; w = target_size; h = h * scale; }
    else { scale = (float)target_size / h; h = target_size; w = w * scale; }
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);
    const int max_stride = 32;
    const int wpad = target_size - w;
    const int hpad = target_size - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);
    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in_pad.substract_mean_normalize(0, norm_vals);
    ncnn::Extractor ex = net.create_extractor();
    ex.input(input_blob_name.c_str(), in_pad);
    ncnn::Mat out;
    ex.extract(output_blob_name.c_str(), out);
    ncnn::Mat out_t;
    common::transpose(out, out_t);
    std::vector<common::Object> proposals;
    std::vector<int> strides = {8, 16, 32};
    generate_proposals(out_t, strides, in_pad, prob_threshold, proposals);
    qsort_descent_inplace(proposals);
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);
    int count = picked.size();
    objects.resize(count);
    for (int i = 0; i < count; i++) {
        objects[i] = proposals[picked[i]];
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);
        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }
}

} // namespace yolo_ncnn

void test_yolov11_ncnn()
{
    std::string imagepath = "../data/traffic_road.jpg";

    cv::Mat m = cv::imread(imagepath, 1);

    std::vector<common::Object> objects;
    yolo_ncnn::detect(m, objects, 0.25f, 0.45f, 640);

    static const char* class_names[] = {
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

    static cv::Scalar colors[] = {
            cv::Scalar(244, 67, 54),
            cv::Scalar(233, 30, 99),
            cv::Scalar(156, 39, 176),
            cv::Scalar(103, 58, 183),
            cv::Scalar(63, 81, 181),
            cv::Scalar(33, 150, 243),
            cv::Scalar(3, 169, 244),
            cv::Scalar(0, 188, 212),
            cv::Scalar(0, 150, 136),
            cv::Scalar(76, 175, 80),
            cv::Scalar(139, 195, 74),
            cv::Scalar(205, 220, 57),
            cv::Scalar(255, 235, 59),
            cv::Scalar(255, 193, 7),
            cv::Scalar(255, 152, 0),
            cv::Scalar(255, 87, 34),
            cv::Scalar(121, 85, 72),
            cv::Scalar(158, 158, 158),
            cv::Scalar(96, 125, 139)
    };

    cv::Mat image = m.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const common::Object& obj = objects[i];

        const cv::Scalar& color = colors[i % 19];

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(image, obj.rect, color);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    cv::imwrite("../data/yolov11n_traffic_road.jpg",image);
    cv::imshow("image", image);
    cv::waitKey(0);
}