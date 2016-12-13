
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "corner.h"
#include "fast_cuda.h"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/core/core.hpp>



int main(int argc, char* argv[])
{
	std::string input_file;
	std::string output_file = "cornerDetection.png";

	if (argc > 1)
	{
		switch(argc)
		{
		case 2:
			input_file = std::string(argv[1]);
			break;
		default:
			std::cerr << "Bazzinga";
			exit(1);
		}
	}
	else {
		std::cout << "Input the image" << std::endl;
		std::cin >> input_file;
		input_file = "/Users/macbookpro/Downloads/image/" + input_file;
	}

	cv::Mat image;
	image = cv::imread(input_file.c_str(), CV_LOAD_IMAGE_COLOR);
	if (!image.data)
	{
		std::cerr << "Couldn't open or find the image" << std::endl;
		exit(1);
	}

    const int threshold=40;
    cv::Mat imageDetect = cv::imread(input_file.c_str(), 0);
	std::vector<cv::KeyPoint> keypoint;
	fast_corner_9(imageDetect,keypoint,threshold);

	cv::Mat imageDetected;
	drawKeypoints(image,keypoint,imageDetected,cv::Scalar(0,255,0));
	cv::imwrite(output_file.c_str(), imageDetected);
	cv::Size size1=image.size();
	cv::Size size2=imageDetected.size();
	cv::Mat imageCombine(image.rows,image.cols+imageDetected.cols,CV_8UC3);
	cv::Mat left(imageCombine,cv::Rect(0,0,image.cols,image.rows));
	image.copyTo(left);
	cv::Mat right(imageCombine,cv::Rect(image.cols,0,imageDetected.cols,imageDetected.rows));
	imageDetected.copyTo(right);
	cv::namedWindow( "Display", cv::WINDOW_AUTOSIZE );         // Create a window for display.
	cv::imshow( "Display", imageCombine );                   // Show our image inside it.
	cv::waitKey(0);                                          // Wait for a keystroke in the window


	return 0;
}
