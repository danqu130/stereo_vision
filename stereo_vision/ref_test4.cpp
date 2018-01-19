#include <string>
#include <stdio.h>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "V4L2Capture.cpp"

// #define USE_DISPLAY
#define SHOW_DATA 1
#define FRAME_WIDTH 2560
#define FRAME_HEIGHT 960
#define RESIZE_WIDTH 640
#define RESIZE_HEIGHT 480
#define LEFT_WINDOW_NAME "left"
#define RIGHT_WINDOW_NAME "right"

int stereoPreFilterSize = 9;
int stereoPreFilterCap = 31;
int stereoDispWindowSize = 21;
int stereoNumDisparities = 64;
int stereoDispTextureThreshold = 10;
int stereoDispUniquenessRatio = 5;
int _threshold, blobArea;
bool left_mouse = false;


cv::Mat cameraMatrix[2], distCoeffs[2], cameraData;
cv::Mat R, T, E, F;
cv::Mat R1, R2, P1, P2, Q;
cv::Mat img1, img2, rgb, thres_img, blobs_img, img_detect, real_disparity;
cv::Mat img1_rectified, img2_rectified,  disparityMat, view_disparityMat, depthMat, pairMat;
std::vector<cv::Mat> rgb_detect, rgb_detect_r;
cv::Mat r_detect, g_detect, b_detect, r_detect_r, g_detect_r, b_detect_r;

cv::Ptr<cv::StereoBM> sbm = cv::StereoBM::create(16 * 5, 31);

double reprojectionVars[6];

struct ObjectInfo {
    cv::Point       center;     //中心
    double          distance;   //距离
    double          area;       //面积
    cv::Rect        boundRect;  //外接矩形
    cv::RotatedRect minRect;    //最小矩形

    // 定义赋值操作
    void operator = (const ObjectInfo &rhs) {
        center = rhs.center;
        distance = rhs.distance;
        area = rhs.area;
        boundRect = rhs.boundRect;
        minRect = rhs.minRect;
    }

    // 按照距离定义排序规则
    bool operator < (const ObjectInfo &rhs) const { //升序排序时必须写的函数
        return distance < rhs.distance;
    }
    bool operator >(const ObjectInfo &rhs) const { //降序排序时必须写的函数
        return distance > rhs.distance;
    }

};

std::vector<ObjectInfo>objectinfo;

void stereoCorrelation() {
    sbm->setPreFilterSize(stereoPreFilterSize);
    sbm->setPreFilterCap(stereoPreFilterCap);
    sbm->setBlockSize(stereoDispWindowSize);
    sbm->setMinDisparity(0);
    sbm->setNumDisparities(stereoNumDisparities);
    sbm->setTextureThreshold(stereoDispTextureThreshold);
    sbm->setUniquenessRatio(stereoDispUniquenessRatio);
    sbm->setSpeckleWindowSize(200);
    sbm->setSpeckleRange(32);
    sbm->setDisp12MaxDiff(2);
}
void onWindowBarSlide(int pos) {
    stereoDispWindowSize = cvGetTrackbarPos("SADSize", "Stereo Controls");

    if (stereoDispWindowSize < 5) {
        stereoDispWindowSize = 5;
        stereoCorrelation();
    } else if (stereoDispWindowSize % 2 == 0) {
        stereoDispWindowSize += 1;
        stereoCorrelation();
    } else {
        stereoCorrelation();
    }
}

void onTextureBarSlide(int pos) {
    stereoDispTextureThreshold = cvGetTrackbarPos("Texture th", "Stereo Controls");

    if (stereoDispTextureThreshold) {
        stereoCorrelation();
    }
}

void onUniquenessBarSlide(int pos) {
    stereoDispUniquenessRatio = cvGetTrackbarPos("Uniqueness", "Stereo Controls");

    if (stereoDispUniquenessRatio >= 0) {
        stereoCorrelation();
    }
}

void onNumDisparitiesSlide(int pos) {
    stereoNumDisparities = cvGetTrackbarPos("Num.Disp", "Stereo Controls");

    while (stereoNumDisparities % 16 != 0 || stereoNumDisparities == 0) {
        stereoNumDisparities++;
    }

    stereoCorrelation();
}

int getDisparityImage(cv::Mat &disparity, cv::Mat &disparityImage, bool isColor) {
    // 将原始视差数据的位深转换为 8 位
    cv::Mat disp8u;

    if (disparity.depth() != CV_8U) {
        disparity.convertTo(disp8u, CV_8U, 255 / (stereoNumDisparities * 16.));
    } else {
        disp8u = disparity;
    }

    // 转换为伪彩色图像 或 灰度图像
    if (isColor) {
        if (disparityImage.empty() || disparityImage.type() != CV_8UC3) {
            disparityImage = cv::Mat::zeros(disparity.rows, disparity.cols, CV_8UC3);
        }

        for (int y = 0; y < disparity.rows; y++) {
            for (int x = 0; x < disparity.cols; x++) {
                uchar val = disp8u.at<uchar>(y, x);
                uchar r, g, b;

                if (val == 0) {
                    r = g = b = 0;
                } else {
                    r = 255 - val;
                    g = val < 128 ? val * 2 : (uchar)((255 - val) * 2);
                    b = val;
                }

                disparityImage.at<cv::Vec3b>(y, x) = cv::Vec3b(r, g, b);
            }
        }
    } else {
        disp8u.copyTo(disparityImage);
    }

    return 1;
}



void Stereo() {
    unsigned char *yuv422frame = NULL;
    unsigned long yuvframeSize = 0;

    std::string videoDev = "/dev/video1";
    V4L2Capture *vcap = new V4L2Capture(const_cast<char *>(videoDev.c_str()), FRAME_WIDTH, FRAME_HEIGHT);
    vcap->openDevice();
    vcap->initDevice();
    vcap->startCapture();

#ifdef USE_DISPLAY
    cv::namedWindow(LEFT_WINDOW_NAME, CV_WINDOW_AUTOSIZE);
    cv::namedWindow(RIGHT_WINDOW_NAME, CV_WINDOW_AUTOSIZE);
#endif

    CvMat cvmat;
    IplImage *img;
    cv::Mat matimg;
    double t;
    cv::Size imageSize(RESIZE_WIDTH, RESIZE_HEIGHT);
    cv::Rect validRoi[2];
    cv::Mat rmap[2][2];
    cv::Mat frame_l;
    cv::Mat frame_r;

    // cv::Rect validRoi[2];
    cv::FileStorage fs("cam_stereo.yml", cv::FileStorage::READ);

    if (fs.isOpened()) {
        fs["K1"] >> cameraMatrix[0];
        fs["D1"] >> distCoeffs[0];
        fs["K2"] >> cameraMatrix[1];
        fs["D2"] >> distCoeffs[1];
        fs["R"] >> R;
        fs["T"] >> T;
        fs["R1"] >> R1;
        fs["R2"] >> R2;
        fs["P1"] >> P1;
        fs["P2"] >> P2;
        fs["Q"] >> Q;
        fs["CamData"] >> cameraData;
        fs.release();
    } else {
        cout << "Error: can not save the extrinsic parameters\n";
    }

    cv::stereoRectify(cameraMatrix[0], distCoeffs[0],
                      cameraMatrix[1], distCoeffs[1],
                      imageSize, R, T, R1, R2, P1, P2, Q,
                      cv::CALIB_ZERO_DISPARITY, -1, imageSize, &validRoi[0], &validRoi[1]);

    std::cout << validRoi[0].size() << std::endl;
    std::cout << validRoi[1].size() << std::endl;

    //Precompute maps for cv::remap()
    cv::initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
    cv::initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);

    reprojectionVars[0] = cameraMatrix[0].at<double>(0, 0);
    reprojectionVars[1] = cameraMatrix[1].at<double>(0, 0);
    reprojectionVars[2] = (-1) * T.at<double>(0, 0);
    reprojectionVars[3] = cameraData.at<double>(0, 0);
    std::cout << reprojectionVars[3] << std::endl;
    reprojectionVars[4] = cameraData.at<double>(0, 1);
    reprojectionVars[5] = cameraData.at<double>(0, 2);
    std::cout << reprojectionVars[5] << std::endl;

    cv::namedWindow("Stereo Controls");
    cv::resizeWindow("Stereo Controls", 600, 300);
    cvCreateTrackbar("SADSize", "Stereo Controls", &stereoDispWindowSize, 255, onWindowBarSlide);
    cvCreateTrackbar("Texture th", "Stereo Controls", &stereoDispTextureThreshold, 25, onTextureBarSlide);
    cvCreateTrackbar("Uniqueness", "Stereo Controls", &stereoDispUniquenessRatio, 25, onUniquenessBarSlide);
    cvCreateTrackbar("Num.Disp", "Stereo Controls", &stereoNumDisparities, 640, onNumDisparitiesSlide);
    cvCreateTrackbar("Threshold", "Stereo Controls", &_threshold, 300, 0);
    cvCreateTrackbar("Area", "Stereo Controls", &blobArea, 5000, 0);


    cv::namedWindow("Rectified");
    cv::namedWindow("Disparity Map");

    while (1) {
        t = (double)cvGetTickCount();
        vcap->getFrame((void **)&yuv422frame, (size_t *)&yuvframeSize);
        cvmat = cv::Mat(IMAGEHEIGHT, IMAGEWIDTH, CV_8UC3, (void *)yuv422frame);

        // decode
        img = cvDecodeImage(&cvmat, 1);

        if (!img) {
            std::cout << "DecodeImage error!" << std::endl;
            continue;
        }

        matimg = cv::cvarrToMat(img);
        // cv::imshow("mat", matimg);
        frame_l = matimg(cv::Range(0, FRAME_HEIGHT), cv::Range(0, FRAME_WIDTH / 2));
        frame_r = matimg(cv::Range(0, FRAME_HEIGHT), cv::Range(FRAME_WIDTH / 2, FRAME_WIDTH));
        cv::resize(frame_l, frame_l, cv::Size(RESIZE_WIDTH, RESIZE_HEIGHT));
        cv::resize(frame_r, frame_r, cv::Size(RESIZE_WIDTH, RESIZE_HEIGHT));
#ifdef USE_DISPLAY
        cv::imshow(LEFT_WINDOW_NAME, frame_l);
        cv::imshow(RIGHT_WINDOW_NAME, frame_r);
#endif

        if (frame_l.empty() || frame_r.empty()) {
            continue;
        }

        stereoCorrelation();

        cvReleaseImage(&img);
        vcap->backFrame();

        if ((cvWaitKey(1) & 255) == 27) {
            exit(0);
        }

        t = (double)cvGetTickCount() - t;
        printf("Used time is %g ms, fps is %g\n", (t / (cvGetTickFrequency() * 1000)), 1000 / (t / (cvGetTickFrequency() * 1000)));
    }

    vcap->stopCapture();
    vcap->freeBuffers();
    vcap->closeDevice();
}

int main() {
    Stereo();
    return 0;
}
