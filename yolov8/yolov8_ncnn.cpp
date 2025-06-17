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
    std::string output_blob_name = "216";
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