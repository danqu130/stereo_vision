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
#define FRAME_WIDTH 2560
#define FRAME_HEIGHT 960
#define RESIZE_WIDTH 640
#define RESIZE_HEIGHT 480
#define LEFT_WINDOW_NAME "left"
#define RIGHT_WINDOW_NAME "right"

bool left_mouse = false;
int pic_info[4];

static void onMouse(int event, int x, int y, int /*flags*/, void* param)
{
	cv::Mat& image = *(cv::Mat*) param;  

	if (event == cv::EVENT_LBUTTONDOWN)
	{
		pic_info[0] = x;
		pic_info[1] = y;
		std::cout << "x:" << pic_info[0] << "y:" << pic_info[1] << std::endl;
		left_mouse = true;
	}

	else if (event == CV_EVENT_LBUTTONUP)
	{
		left_mouse = false;
	}

	else if ((event == CV_EVENT_MOUSEMOVE) && (left_mouse == true))
	{
	}
}


int getPointClouds(cv::Mat& disparity, cv::Mat& pointClouds, cv::Mat& Q)
{
	if (disparity.empty())
	{
		return 0;
	}

	//计算生成三维点云
	//  cv::reprojectImageTo3D(disparity, pointClouds, m_Calib_Mat_Q, true);

	cv::reprojectImageTo3D(disparity, pointClouds, Q, true);

	pointClouds *= 1.6;

	for (int y = 0; y < pointClouds.rows; ++y)
	{
		for (int x = 0; x < pointClouds.cols; ++x)
		{
			cv::Point3f point = pointClouds.at<cv::Point3f>(y, x);
			point.y = -point.y;
			pointClouds.at<cv::Point3f>(y, x) = point;
		}
	}

	return 1;
}

void detectDistance(cv::Mat& pointCloud)
{
	if (pointCloud.empty())
	{
		return;
	}

	// 提取深度图像
	std::vector<cv::Mat> xyzSet;
	cv::split(pointCloud, xyzSet);
	cv::Mat depth;
	xyzSet[2].copyTo(depth);

	// 根据深度阈值进行二值化处理
	// double maxVal = 0, minVal = 0;
	// cv::Mat depthThresh = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);
	// cv::minMaxLoc(depth, &minVal, &maxVal);
	// double thrVal = minVal * 1.5;
	// cv::threshold(depth, depthThresh, thrVal, 255, CV_THRESH_BINARY_INV);
	// depthThresh.convertTo(depthThresh, CV_8UC1);
	//imageDenoising(depthThresh, 3);

	double  distance = depth.at<float>(pic_info[0], pic_info[1]);
	cout << "distance:" << distance << endl;
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
    cv::Mat left_img;
    cv::Mat right_img;
    double t;

    cv::Mat cameraMatrix[2], distCoeffs[2], cameraData;
	cv::Mat R, T, E, F;
    cv::Mat R1, R2, P1, P2, Q;
    cv::Rect validRoi[2];
    cv::Size imageSize(RESIZE_WIDTH, RESIZE_HEIGHT);
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

	// OpenCV can handle left-right
    // or up-down camera arrangements
    bool isVerticalStereo = fabs(P2.at<double>(1, 3)) > fabs(P2.at<double>(0, 3));

	// COMPUTE AND DISPLAY RECTIFICATION
    cv::Mat rmap[2][2];
    // IF BY CALIBRATED (BOUGUET'S METHOD)
	//Precompute maps for cv::remap()
    cv::initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
    cv::initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);

	cv::Mat canvas;
    cv::Mat pointClouds;
    double sf;
    int w, h;

	if (!isVerticalStereo) {
        sf = 600. / MAX(imageSize.width, imageSize.height);
        w = cvRound(imageSize.width * sf);
        h = cvRound(imageSize.height * sf);
        canvas.create(h, w * 2, CV_8UC3);
    } else {
        sf = 300. / MAX(imageSize.width, imageSize.height);
        w = cvRound(imageSize.width * sf);
        h = cvRound(imageSize.height * sf);
        canvas.create(h * 2, w, CV_8UC3);
    }

    cv::Mat imgLeft, imgRight;
	int ndisparities = 16 * 5;   /**< Range of disparity */
    int SADWindowSize = 31; /**< Size of the block window. Must be odd */
    cv::Ptr<cv::StereoBM> sbm = cv::StereoBM::create(ndisparities, SADWindowSize);
	sbm->setPreFilterType(CV_STEREO_BM_XSOBEL);  //CV_STEREO_BM_NORMALIZED_RESPONSE或者CV_STEREO_BM_XSOBEL
    sbm->setPreFilterSize(9);
	sbm->setPreFilterCap(31);
	sbm->setBlockSize(15);
	sbm->setMinDisparity(0);
	sbm->setNumDisparities(64);
	sbm->setTextureThreshold(10);
	sbm->setUniquenessRatio(5);
	sbm->setSpeckleWindowSize(100);
	sbm->setSpeckleRange(32);

	cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0, 64, 7,
                                              10 * 7 * 7,
                                              40 * 7 * 7,
                                              1, 63, 10, 100, 32, cv::StereoSGBM::MODE_SGBM);

	cv::Mat rimg, cimg;
    cv::Mat Mask;

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
        cv::resize(left_img, left_img, cv::Size(RESIZE_WIDTH, RESIZE_HEIGHT));
        cv::resize(right_img, right_img, cv::Size(RESIZE_WIDTH, RESIZE_HEIGHT));
#ifdef USE_DISPLAY
        cv::imshow(LEFT_WINDOW_NAME, left_img);
        cv::imshow(RIGHT_WINDOW_NAME, right_img);
#endif

		if (left_img.empty() || right_img.empty()) {
            continue;
        }

        // 单目相机矫正
		cv::remap(left_img, rimg, rmap[0][0], rmap[0][1], cv::INTER_LINEAR);
		rimg.copyTo(cimg);
        cv::Mat canvasPart1 = !isVerticalStereo ? canvas(cv::Rect(w * 0, 0, w, h)) : canvas(cv::Rect(0, h * 0, w, h));
        cv::resize(cimg, canvasPart1, canvasPart1.size(), 0, 0, cv::INTER_AREA);
        cv::Rect vroi1 = cv::Rect(cvRound(validRoi[0].x * sf), cvRound(validRoi[0].y * sf),
                   cvRound(validRoi[0].width * sf), cvRound(validRoi[0].height * sf));
		cv::remap(right_img, rimg, rmap[1][0], rmap[1][1], cv::INTER_LINEAR);
        rimg.copyTo(cimg);
        cv::Mat canvasPart2 = !isVerticalStereo ? canvas(cv::Rect(w * 1, 0, w, h)) : canvas(cv::Rect(0, h * 1, w, h));
        cv::resize(cimg, canvasPart2, canvasPart2.size(), 0, 0, cv::INTER_AREA);
        cv::Rect vroi2 = cv::Rect(cvRound(validRoi[1].x * sf), cvRound(validRoi[1].y * sf),
                          cvRound(validRoi[1].width * sf), cvRound(validRoi[1].height * sf));

		cv::Rect vroi = vroi1 & vroi2;
		
        imgLeft = canvasPart1(vroi).clone();
        imgRight = canvasPart2(vroi).clone();
        
		cv::rectangle(canvasPart1, vroi1, cv::Scalar(0, 0, 255), 3, 8);
        cv::rectangle(canvasPart2, vroi2, cv::Scalar(0, 0, 255), 3, 8);

        // 双目矫正对齐
		if (!isVerticalStereo)
            for (int j = 0; j < canvas.rows; j += 32) {
                cv::line(canvas, cv::Point(0, j), cv::Point(canvas.cols, j), cv::Scalar(0, 255, 0), 1, 8);
            }
        else
            for (int j = 0; j < canvas.cols; j += 32) {
                cv::line(canvas, cv::Point(j, 0), cv::Point(j, canvas.rows), cv::Scalar(0, 255, 0), 1, 8);
            }

		
        cv::cvtColor(imgLeft, imgLeft, CV_BGR2GRAY);
        cv::cvtColor(imgRight, imgRight, CV_BGR2GRAY);

		//-- And create the image in which we will save our disparities
        cv::Mat imgDisparity16S = cv::Mat(imgLeft.rows, imgLeft.cols, CV_16S);
        cv::Mat imgDisparity8U = cv::Mat(imgLeft.rows, imgLeft.cols, CV_8UC1);
        cv::Mat sgbmDisp16S = cv::Mat(imgLeft.rows, imgLeft.cols, CV_16S);
        cv::Mat sgbmDisp8U = cv::Mat(imgLeft.rows, imgLeft.cols, CV_8UC1);

		if (imgLeft.empty() || imgRight.empty()) {
            std::cout << " --(!) Error reading images " << std::endl;
            continue;
        }

		sbm->compute(imgLeft, imgRight, imgDisparity16S);

		imgDisparity16S.convertTo(imgDisparity8U, CV_8UC1, 255.0 / 1000.0);
        cv::compare(imgDisparity16S, 0, Mask, cv::CMP_GE);
        cv::Mat disparity;
        imgDisparity8U.copyTo(disparity, Mask);
        
        getPointClouds(disparity, pointClouds, Q);
		detectDistance(pointClouds);


        cv::Mat disparityShow;
        cv::applyColorMap(imgDisparity8U, imgDisparity8U, cv::COLORMAP_HSV);
        imgDisparity8U.copyTo(disparityShow, Mask);



        // sgbm->compute(imgLeft, imgRight, sgbmDisp16S);

        // sgbmDisp16S.convertTo(sgbmDisp8U, CV_8UC1, 255.0 / 1000.0);
        // cv::compare(sgbmDisp16S, 0, Mask, cv::CMP_GE);

        // cv::Mat disparity;
        // sgbmDisp8U.copyTo(disparity, Mask);
        // getPointClouds(disparity, pointClouds, Q);
		// detectDistance(pointClouds);

        // cv::Mat sgbmDisparityShow;
        // cv::applyColorMap(sgbmDisp8U, sgbmDisp8U, cv::COLORMAP_HSV);
        // sgbmDisp8U.copyTo(sgbmDisparityShow, Mask);
        // std::cout << "imgDisparity8U.center: " << imgDisparity8U.at<float>(RESIZE_WIDTH/2, RESIZE_HEIGHT/2);
        // std::cout << "sgbmDisp8U.center: " << sgbmDisp8U.at<float>(RESIZE_WIDTH/2, RESIZE_HEIGHT/2) << std::endl;

        cv::namedWindow("rectified");
		cv::setMouseCallback("rectified", onMouse, &canvas);


		cv::imshow("bmDisparity", disparityShow);
        // cv::imshow("sgbmDisparity", sgbmDisparityShow);
        cv::imshow("rectified", canvas);


        cvReleaseImage(&img);
        vcap->backFrame();

        if ((cvWaitKey(1) & 255) == 27) {
            exit(0);
        }

        t = (double)cvGetTickCount() - t;
        // printf("Used time is %g ms, fps is %g\n", (t / (cvGetTickFrequency() * 1000)), 1000 / (t / (cvGetTickFrequency() * 1000)));
    }
    vcap->stopCapture();
    vcap->freeBuffers();
    vcap->closeDevice();
}

int main() {
    Stereo();
    return 0;
}
