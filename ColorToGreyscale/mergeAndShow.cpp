#include <iostream>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <opencv2/opencv.hpp>



void mergeAndShow(cv::Mat& imageRGB, cv::Mat& imageGrey)
{
	cv::Mat mergeImage;
	std::vector<cv::Mat> channels(3, imageGrey);
	cv::merge(channels, mergeImage);
	cv::Size sizeColor = imageRGB.size();
	cv::Size sizeGrey = imageGrey.size();
	cv::Mat imageCombine(imageRGB.rows, imageRGB.cols + imageGrey.cols, CV_8UC3);
	cv::Mat left(imageCombine, cv::Rect(0, 0, imageRGB.cols, imageRGB.rows));
	imageRGB.copyTo(left);
	cv::Mat right(imageCombine, cv::Rect(imageRGB.cols, 0, imageGrey.cols, imageGrey.rows));
	mergeImage.copyTo(right);
	cv::namedWindow("Display", cv::WINDOW_AUTOSIZE );
	cv::imshow( "Display", imageCombine);
	cv::waitKey(0);
}
