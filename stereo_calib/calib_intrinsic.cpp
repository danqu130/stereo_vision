#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include "popt_pp.h"

std::vector< std::vector<cv::Point3f> > object_points;
std::vector< std::vector<cv::Point2f> > image_points;
std::vector<cv::Point2f> corners;

cv::Mat img, gray;
cv::Size im_size;

void setup_calibration(int board_width, int board_height, int num_imgs,
                       float square_size, char *imgs_directory, char *imgs_filename,
                       char *extension) {
    cv::Size board_size = cv::Size(board_width, board_height);
    int board_n = board_width * board_height;

    for (int k = 1; k <= num_imgs; k++) {
        char img_file[100];
        sprintf(img_file, "%s%s%d.%s", imgs_directory, imgs_filename, k, extension);
        img = cv::imread(img_file, CV_LOAD_IMAGE_COLOR);
        cv::cvtColor(img, gray, CV_BGR2GRAY);

        bool found = false;
        found = cv::findChessboardCorners(img, board_size, corners,
                                          CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);

        if (found) {
            cv::cornerSubPix(gray, corners, cv::Size(5, 5), cv::Size(-1, -1),
                         cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
            cv::drawChessboardCorners(gray, board_size, corners, found);
        }

        std::vector<cv::Point3f> obj;

        for (int i = 0; i < board_height; i++)
            for (int j = 0; j < board_width; j++) {
                obj.push_back(cv::Point3f((float)j * square_size, (float)i * square_size, 0));
            }

        if (found) {
            std::cout << k << ". Found corners!" << std::endl;
            image_points.push_back(corners);
            object_points.push_back(obj);
        }
    }
}

double computeReprojectionErrors(const std::vector<std::vector<cv::Point3f> > &objectPoints,
                                 const std::vector<std::vector<cv::Point2f> > &imagePoints,
                                 const std::vector<cv::Mat> &rvecs, const std::vector<cv::Mat> &tvecs,
                                 const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs) {
    std::vector<cv::Point2f> imagePoints2;
    int i, totalPoints = 0;
    double totalErr = 0, err;
    std::vector<float> perViewErrors;
    perViewErrors.resize(objectPoints.size());

    for (i = 0; i < (int)objectPoints.size(); ++i) {
        cv::projectPoints(cv::Mat(objectPoints[i]), rvecs[i], tvecs[i], cameraMatrix,
                      distCoeffs, imagePoints2);
        err = cv::norm(cv::Mat(imagePoints[i]), cv::Mat(imagePoints2), CV_L2);
        int n = (int)objectPoints[i].size();
        perViewErrors[i] = (float)std::sqrt(err * err / n);
        totalErr += err * err;
        totalPoints += n;
    }

    return std::sqrt(totalErr / totalPoints);
}

int main(int argc, char const **argv) {
    int board_width, board_height, num_imgs;
    float square_size;
    char *imgs_directory;
    char *imgs_filename;
    char *out_file;
    char *extension;

    static struct poptOption options[] = {
        {"board_width", 'w', POPT_ARG_INT, &board_width, 0, "Checkerboard width", "NUM"},
        {"board_height", 'h', POPT_ARG_INT, &board_height, 0, "Checkerboard height", "NUM"},
        {"num_imgs", 'n', POPT_ARG_INT, &num_imgs, 0, "Number of checkerboard images", "NUM"},
        {"square_size", 's', POPT_ARG_FLOAT, &square_size, 0, "Size of checkerboard square", "NUM"},
        {"imgs_directory", 'd', POPT_ARG_STRING, &imgs_directory, 0, "Directory containing images", "STR"},
        {"imgs_filename", 'i', POPT_ARG_STRING, &imgs_filename, 0, "Image filename", "STR"},
        {"extension", 'e', POPT_ARG_STRING, &extension, 0, "Image extension", "STR"},
        {"out_file", 'o', POPT_ARG_STRING, &out_file, 0, "Output calibration filename (YML)", "STR"},
        POPT_AUTOHELP{NULL, 0, 0, NULL, 0, NULL, NULL}
    };

    POpt popt(NULL, argc, argv, options, 0);
    int c;

    while ((c = popt.getNextOpt()) >= 0) {
    }

    setup_calibration(board_width, board_height, num_imgs, square_size,
                      imgs_directory, imgs_filename, extension);

    printf("Starting Calibration\n");
    cv::Mat K;
    cv::Mat D;
    std::vector<cv::Mat> rvecs, tvecs;
    int flag = 0;
    flag |= CV_CALIB_FIX_K4;
    flag |= CV_CALIB_FIX_K5;
    calibrateCamera(object_points, image_points, img.size(), K, D, rvecs, tvecs, flag);

    std::cout << "Calibration error: " << computeReprojectionErrors(object_points, image_points, rvecs, tvecs, K, D) << std::endl;

    cv::FileStorage fs(out_file, cv::FileStorage::WRITE);
    fs << "K" << K;
    fs << "D" << D;
    fs << "board_width" << board_width;
    fs << "board_height" << board_height;
    fs << "square_size" << square_size;
    printf("Done Calibration\n");

    return 0;
}
