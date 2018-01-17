#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "V4L2Capture.cpp"

#define USE_DISPLAY
#define FRAME_WIDTH 2560
#define FRAME_HEIGHT 960
#define LEFT_WINDOW_NAME "left"
#define RIGHT_WINDOW_NAME "right"

void VideoSegmentation() {
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
    cv::Mat left_img;
    cv::Mat right_img;
    double t;

    while (1) {
        t = (double)cvGetTickCount();
        vcap->getFrame((void **)&yuv422frame, (size_t *)&yuvframeSize);
        cvmat = cv::Mat(IMAGEHEIGHT, IMAGEWIDTH, CV_8UC3, (void *)yuv422frame);

        // decode
        img = cvDecodeImage(&cvmat, 1);

        if (!img) {
            std::cout << "DecodeImage error!" << std::endl;
        }

        matimg = cv::cvarrToMat(img);
        // cv::imshow("mat", matimg);
        left_img = matimg(cv::Range(0, FRAME_HEIGHT), cv::Range(0, FRAME_WIDTH / 2));
        right_img = matimg(cv::Range(0, FRAME_HEIGHT), cv::Range(FRAME_WIDTH / 2, FRAME_WIDTH));

#ifdef USE_DISPLAY
        cv::imshow(LEFT_WINDOW_NAME, left_img);
        cv::imshow(RIGHT_WINDOW_NAME, right_img);
#endif


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
    VideoSegmentation();
    return 0;
}
