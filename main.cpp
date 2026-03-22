#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

#include "benchmark_mac.hpp"

using namespace std;
using namespace cv;
using namespace cv::dnn;

using steady_clock = chrono::steady_clock;
using system_clock = chrono::system_clock;

// CORRECT INDICES FOR 1-CLASS YOLOv8-SEG
// Columns 0-3: box | Column 4: class confidence | Columns 5-36: 32 mask coefficients
static const int CONF_IDX = 4;
static const int MASK_START_IDX = 5;
// Stricter thresholds: one clean face, suppress overlapping boxes
static const float CONF_THRESHOLD = 0.50f;  // 50% confidence
static const float NMS_THRESHOLD = 0.25f;   // Strict: kill overlapping boxes
static const int MAX_DETECTIONS = 10;       // Draw up to 10 faces

static void printUsage(const char* prog) {
    cerr << "Usage: " << prog << " [options]\n"
         << "  --model PATH          ONNX model (default: yolov8s-seg.onnx)\n"
         << "  --benchmark           Show on-screen metrics (disk, RAM, DNN ms, CPU proxy)\n"
         << "  --benchmark-log FILE  Append CSV metrics every ~1s (for spreadsheets)\n"
         << "  --bench-frames N      Exit after N frames (steady benchmark)\n"
         << "  -h, --help            This help\n"
         << "Press ESC to quit.\n";
}

static string formatMb(double bytes) {
    ostringstream o;
    o << fixed << setprecision(2) << (bytes / (1024.0 * 1024.0)) << " MB";
    return o.str();
}

int main(int argc, char** argv) {
    string modelPath = "yolov8s-seg.onnx";
    bool showBenchOverlay = false;
    string benchLogPath;
    int benchFrames = -1;

    for (int i = 1; i < argc; ++i) {
        string a = argv[i];
        if (a == "-h" || a == "--help") {
            printUsage(argv[0]);
            return 0;
        }
        if (a == "--benchmark") {
            showBenchOverlay = true;
            continue;
        }
        if (a == "--benchmark-log" && i + 1 < argc) {
            benchLogPath = argv[++i];
            continue;
        }
        if (a == "--bench-frames" && i + 1 < argc) {
            benchFrames = atoi(argv[++i]);
            continue;
        }
        if (a == "--model" && i + 1 < argc) {
            modelPath = argv[++i];
            continue;
        }
        cerr << "Unknown argument: " << a << endl;
        printUsage(argv[0]);
        return 1;
    }

    const bool collectMetrics = showBenchOverlay || !benchLogPath.empty() || benchFrames > 0;

    Net net = readNetFromONNX(modelPath);

    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "CRITICAL: Could not open Mac webcam." << endl;
        return -1;
    }
    cap.set(CAP_PROP_BUFFERSIZE, 1);

    const int64_t onnxBytes = getFileSizeBytes(modelPath);
    const unsigned ncpu = getLogicalCpuCount();

    RollingLatencyMs inferStats(120);
    BenchmarkSampler cpuSampler;
    size_t peakRssBytes = 0;
    double lastCpuProcPct = 0.0;
    size_t lastRssBytes = 0;

    ofstream benchLog;
    bool csvHeaderWritten = false;
    if (!benchLogPath.empty()) {
        benchLog.open(benchLogPath.c_str(), ios::app);
        if (!benchLog) {
            cerr << "Could not open --benchmark-log file: " << benchLogPath << endl;
            return 1;
        }
    }

    if (collectMetrics) {
        cpuSampler.reset();
        lastRssBytes = getProcessResidentBytes();
        peakRssBytes = lastRssBytes;
    }

    steady_clock::time_point lastSampleWall = steady_clock::now();

    Mat frame;
    cout << "--- MAC M2 TERNARY ENGINE: BOXES + MASKS RUNNING ---" << endl;
    if (collectMetrics) {
        cout << "Benchmark: ONNX disk=";
        if (onnxBytes >= 0) {
            cout << formatMb(static_cast<double>(onnxBytes));
        } else {
            cout << "N/A";
        }
        cout << " | logical CPUs=" << ncpu << endl;
    }
    cout << "Press ESC to quit." << endl;

    int frameIndex = 0;

    while (cap.read(frame)) {
        Mat blob;
        blobFromImage(frame, blob, 1.0 / 255.0, Size(640, 640), Scalar(), true, false);
        net.setInput(blob);

        vector<Mat> outputs;
        const auto t0 = chrono::high_resolution_clock::now();
        net.forward(outputs, net.getUnconnectedOutLayersNames());
        const auto t1 = chrono::high_resolution_clock::now();
        if (collectMetrics) {
            const double inferMs = chrono::duration<double, milli>(t1 - t0).count();
            inferStats.push(inferMs);
        }

        // ONNX may swap output order: one is Predictions [1,37,8400], one is Prototypes [1,32,160,160]
        Mat predictions, proto;
        if (outputs[0].dims == 3) {
            predictions = outputs[0];
            proto = outputs[1];
        } else {
            predictions = outputs[1];
            proto = outputs[0];
        }

        // Prototype masks [1, 32, 160, 160] -> [32, 25600] for matrix multiply
        Mat proto_flat = proto.reshape(1, 32);

        int dimensions = predictions.size[1];   // 37
        int num_proposals = predictions.size[2]; // 8400

        predictions = predictions.reshape(1, dimensions);
        Mat data;
        transpose(predictions, data);

        vector<Rect> boxes;
        vector<float> confidences;
        vector<Mat> mask_coeffs;

        float x_factor = frame.cols / 640.0f;
        float y_factor = frame.rows / 640.0f;

        // 1. Scan for valid faces (confidence at index 4)
        for (int i = 0; i < num_proposals; ++i) {
            float* row = data.ptr<float>(i);
            float conf = row[CONF_IDX];

            if (conf >= CONF_THRESHOLD) {
                float cx = row[0], cy = row[1], w = row[2], h = row[3];

                int left   = (int)((cx - 0.5f * w) * x_factor);
                int top    = (int)((cy - 0.5f * h) * y_factor);
                int width  = (int)(w * x_factor);
                int height = (int)(h * y_factor);

                boxes.push_back(Rect(left, top, width, height));
                confidences.push_back(conf);

                // Extract the 32 mask coefficients (indices 5 to 36)
                Mat coeff(1, 32, CV_32F);
                for (int c = 0; c < 32; ++c) {
                    coeff.at<float>(0, c) = row[MASK_START_IDX + c];
                }
                mask_coeffs.push_back(coeff);
            }
        }

        // 2. Non-Maximum Suppression
        vector<int> indices;
        NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD, indices);

        // 3. Draw boxes and masks (hard cap: only the best face)
        int drawCount = 0;
        for (int idx : indices) {
            if (drawCount >= MAX_DETECTIONS) break;

            Rect box = boxes[idx];

            box.x = max(0, box.x);
            box.y = max(0, box.y);
            box.width = min(frame.cols - box.x, box.width);
            box.height = min(frame.rows - box.y, box.height);
            if (box.width <= 0 || box.height <= 0) continue;

            // Coefficients [1, 32] * Prototypes [32, 25600] = mask [1, 25600]
            Mat mask_mat = mask_coeffs[idx] * proto_flat;
            mask_mat = mask_mat.reshape(1, 160);  // 160x160 grid

            // Sigmoid: 1 / (1 + exp(-x))
            Mat neg;
            multiply(mask_mat, -1.0, neg);
            exp(neg, mask_mat);
            add(mask_mat, Scalar(1.0), mask_mat);
            divide(1.0, mask_mat, mask_mat);

            // Resize mask to frame size, then crop to box
            Mat resized_mask;
            resize(mask_mat, resized_mask, Size(frame.cols, frame.rows));

            Mat roi_mask = resized_mask(box);
            Mat binary_mask;
            threshold(roi_mask, binary_mask, 0.5, 1.0, THRESH_BINARY);
            binary_mask.convertTo(binary_mask, CV_8UC1);

            // Segmentation only: translucent green over the mask region (no bounding box, no label)
            Mat color_mask = frame(box).clone();
            color_mask.setTo(Scalar(0, 255, 0), binary_mask);
            addWeighted(frame(box), 0.5, color_mask, 0.5, 0.0, frame(box));

            drawCount++;
        }

        ++frameIndex;

        if (collectMetrics) {
            const steady_clock::time_point nowWall = steady_clock::now();
            const double sinceSampleSec = chrono::duration<double>(nowWall - lastSampleWall).count();
            if (sinceSampleSec >= 1.0) {
                lastSampleWall = nowWall;
                lastRssBytes = getProcessResidentBytes();
                if (lastRssBytes > peakRssBytes) {
                    peakRssBytes = lastRssBytes;
                }
                lastCpuProcPct = cpuSampler.pollCpuPercent();

                if (benchLog.is_open()) {
                    if (!csvHeaderWritten) {
                        benchLog << "unix_ms,onnx_bytes,rss_bytes,rss_peak_bytes,"
                                 << "infer_mean_ms,infer_p50_ms,infer_p95_ms,"
                                 << "cpu_proc_pct,cpu_per_core_est\n";
                        csvHeaderWritten = true;
                    }
                    const auto unixMs = chrono::duration_cast<chrono::milliseconds>(
                                            system_clock::now().time_since_epoch())
                                            .count();
                    const double perCoreEst =
                        ncpu > 0 ? lastCpuProcPct / static_cast<double>(ncpu) : lastCpuProcPct;
                    benchLog << unixMs << ','
                             << onnxBytes << ','
                             << lastRssBytes << ','
                             << peakRssBytes << ','
                             << fixed << setprecision(3)
                             << inferStats.mean() << ','
                             << inferStats.percentile(50.0) << ','
                             << inferStats.percentile(95.0) << ','
                             << setprecision(2)
                             << lastCpuProcPct << ','
                             << perCoreEst << '\n';
                    benchLog.flush();
                }
            }
        }

        if (showBenchOverlay && collectMetrics) {
            ostringstream line1, line2, line3;
            line1 << "ONNX disk: ";
            if (onnxBytes >= 0) {
                line1 << formatMb(static_cast<double>(onnxBytes));
            } else {
                line1 << "N/A";
            }
            line1 << " | RSS: " << formatMb(static_cast<double>(lastRssBytes))
                  << " (peak " << formatMb(static_cast<double>(peakRssBytes)) << ")";

            line2 << "DNN forward: mean " << fixed << setprecision(2) << inferStats.mean()
                  << " ms | p50 " << inferStats.percentile(50.0)
                  << " | p95 " << inferStats.percentile(95.0) << " ms";

            const double perCoreEst =
                ncpu > 0 ? lastCpuProcPct / static_cast<double>(ncpu) : lastCpuProcPct;
            line3 << "CPU proxy: " << setprecision(1) << lastCpuProcPct << "% proc (~"
                  << perCoreEst << "%/core est) | nCPU=" << ncpu;

            const string s1 = line1.str();
            const string s2 = line2.str();
            const string s3 = line3.str();

            const int margin = 10;
            const Scalar fg(0, 255, 0);
            const Scalar bg(0, 0, 0);
            auto drawLine = [&](const string& s, int y, int thickOutline, int thickFg) {
                putText(frame, s, Point(margin, y), FONT_HERSHEY_SIMPLEX, 0.55, bg, thickOutline, LINE_AA);
                putText(frame, s, Point(margin, y), FONT_HERSHEY_SIMPLEX, 0.55, fg, thickFg, LINE_AA);
            };
            int y = margin + 18;
            drawLine(s1, y, 4, 2);
            y += 22;
            drawLine(s2, y, 4, 2);
            y += 22;
            drawLine(s3, y, 4, 2);
        }

        imshow("M2 Ternary Vision", frame);
        if (waitKey(1) == 27) break;

        if (benchFrames > 0 && frameIndex >= benchFrames) {
            break;
        }
    }

    if (benchLog.is_open()) {
        benchLog.close();
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
