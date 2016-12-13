#include <iostream>
#include <string>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include "utils.h"
#include "timer.h"


void constructFilter(float* const h_filter, const int blurKernelWidth);

void doBlurImage(uchar4* const d_inputImageRGBA,
		uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols, const size_t numPixels,
		uchar* d_redBlurred, uchar* d_greenBlurred, uchar* d_blueBlurred, float* d_filter);
/*******  Begin main *********/

int main(int argc, char **argv) {
	std::string input_file;
	std::string output_file = "processedImage.png";

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
	const size_t numRows = image.rows;
	const size_t numCols = image.cols;
	const size_t numPixels = numRows * numCols;

	cv::Mat imageInputRGBA;
	cv::Mat imageOutputRGBA;
	uchar4 *h_inputImageRGBA, *d_inputImageRGBA;
	uchar4 *h_outputImageRGBA, *d_outputImageRGBA;
	uchar  *d_redBlurred, *d_greenBlurred, *d_blueBlurred;


	cv::cvtColor(image, imageInputRGBA, CV_BGR2RGBA);
	imageOutputRGBA.create(image.rows, image.cols, CV_8UC4);
	h_inputImageRGBA = (uchar4*)imageInputRGBA.ptr<uchar4>(0);
	h_outputImageRGBA = (uchar4*)imageOutputRGBA.ptr<uchar4>(0);

	checkCudaErrors(cudaMalloc(&d_inputImageRGBA, sizeof(uchar4) * numPixels));
	checkCudaErrors(cudaMalloc(&d_outputImageRGBA, sizeof(uchar4) * numPixels));
	checkCudaErrors(cudaMemset(d_outputImageRGBA, 0, numPixels * sizeof(uchar4)));
	checkCudaErrors(cudaMemcpy(d_inputImageRGBA, h_inputImageRGBA, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice));
	//allocate memory for blurred channel
	checkCudaErrors(cudaMalloc(&d_redBlurred,    sizeof(uchar) * numPixels));
	checkCudaErrors(cudaMalloc(&d_greenBlurred,  sizeof(uchar) * numPixels));
	checkCudaErrors(cudaMalloc(&d_blueBlurred,   sizeof(uchar) * numPixels));
	checkCudaErrors(cudaMemset(d_redBlurred,   0, sizeof(uchar) * numPixels));
	checkCudaErrors(cudaMemset(d_greenBlurred, 0, sizeof(uchar) * numPixels));
	checkCudaErrors(cudaMemset(d_blueBlurred,  0, sizeof(uchar) * numPixels));

	int filterWidth = 9; int blurKernelWidth = 9;
	float* h_filter = new float[blurKernelWidth * blurKernelWidth];;
	constructFilter(h_filter, blurKernelWidth);
	//	for(int i = 0; i < 81; i++)
	//	{
	//		std::cout << h_filter[i] << " ";
	//	}

	float *d_filter;
	checkCudaErrors(cudaMalloc(&d_filter, sizeof(float)*filterWidth*filterWidth));
	checkCudaErrors(cudaMemcpy(d_filter, h_filter, sizeof(float)*filterWidth*filterWidth, cudaMemcpyHostToDevice));
	GpuTimer timer;
	timer.Start();
	doBlurImage(d_inputImageRGBA, d_outputImageRGBA, numRows, numCols, numPixels,
			d_redBlurred, d_greenBlurred, d_blueBlurred, d_filter);
	timer.Stop();
	checkCudaErrors(cudaMemcpy(h_outputImageRGBA, d_outputImageRGBA, sizeof(uchar4) * numPixels, cudaMemcpyDeviceToHost));

	cv::Mat output(numRows, numCols, CV_8UC4, (void*)h_outputImageRGBA);
	cv::cvtColor(output, output, CV_RGBA2BGR);
	cv::imwrite(output_file.c_str(), output);

	int err = printf("Your code ran in: %f msecs.\n", timer.Elapsed());
	if (err < 0) {
		std::cerr << "Couldn't print timing information!" << std::endl;
		exit(1);
	}
	cv::Size size1=image.size();
	cv::Size size2=output.size();
	cv::Mat imageCombine(image.rows,image.cols+output.cols,CV_8UC3);
	cv::Mat left(imageCombine,cv::Rect(0,0,image.cols,image.rows));
	image.copyTo(left);
	cv::Mat right(imageCombine,cv::Rect(image.cols,0,output.cols,output.rows));
	output.copyTo(right);
	cv::namedWindow( "Display", cv::WINDOW_AUTOSIZE );         // Create a window for display.
	cv::imshow( "Display", imageCombine );                   // Show our image inside it.
	cv::waitKey(0);                                          // Wait


	checkCudaErrors(cudaFree(d_inputImageRGBA));
	checkCudaErrors(cudaFree(d_outputImageRGBA));
	checkCudaErrors(cudaFree(d_redBlurred));
	checkCudaErrors(cudaFree(d_greenBlurred));
	checkCudaErrors(cudaFree(d_blueBlurred));
	checkCudaErrors(cudaFree(d_filter));


	return 0;
}
