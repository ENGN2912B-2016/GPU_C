


#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <cuda_runtime.h>



void preProcess(cv::Mat& image, cv::Mat& imageRGBA,
		int numRows, int numCols, uchar4** h_rgbaImage, uchar** h_greyImage)
{
	cv::Mat imageGrey;
	cv::cvtColor(image, imageRGBA, CV_BGR2RGBA);
	imageGrey.create(numRows, numCols, CV_8UC1);
	*h_greyImage = imageGrey.ptr<uchar>(0);
	*h_rgbaImage = (uchar4*)imageRGBA.ptr<uchar4>(0);
}
