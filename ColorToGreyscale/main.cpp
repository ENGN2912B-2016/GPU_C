
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


void rgb2grey(const uchar4* const d_rgbaImage,
		uchar* const d_greyImage,
		size_t numRows, size_t numCols);
void mergeAndShow(cv::Mat& imageRGB, cv::Mat& imageGrey);
void preProcess(cv::Mat& image, cv::Mat& imageRGBA,
		int numRows, int numCols, uchar4** h_rgbaImage, uchar**  h_greyImage);

int main(int argc, char* argv[])
{
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
	cv::Mat imageRGBA;
	image = cv::imread(input_file.c_str(), CV_LOAD_IMAGE_COLOR);
	if (!image.data)
	{
		std::cerr << "Couldn't open or find the image" << std::endl;
		exit(1);
	}

	size_t numRows = image.rows;
	size_t numCols = image.cols;
	const size_t numPixels = numRows * numCols;

	uchar4	*h_rgbaImage, *d_rgbaImage;
	uchar 	*h_greyImage,*d_greyImage;
	preProcess(image, imageRGBA, numRows, numCols, &h_rgbaImage, &h_greyImage);

	checkCudaErrors(cudaMalloc(&d_rgbaImage, sizeof(uchar4) * numPixels));
	checkCudaErrors(cudaMalloc(&d_greyImage, sizeof(uchar) * numPixels));
	checkCudaErrors(cudaMemset(d_greyImage, 0, numPixels * sizeof(uchar)));
	checkCudaErrors(cudaMemcpy(d_rgbaImage, h_rgbaImage, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice));
	GpuTimer timer;
	timer.Start();
	rgb2grey(d_rgbaImage, d_greyImage, numRows, numCols);
	timer.Stop();
	checkCudaErrors(cudaMemcpy(h_greyImage, d_greyImage, sizeof(uchar) * numPixels, cudaMemcpyDeviceToHost));
	cv::Mat output(numRows, numCols, CV_8UC1, (void*)h_greyImage);
	cv::imwrite(output_file.c_str(), output);

	mergeAndShow(image, output);
	int err = printf("Your code ran in: %f msecs.\n", timer.Elapsed());
	if (err < 0) {
		std::cerr << "Couldn't print timing information!" << std::endl;
		exit(1);
	}

	cudaFree(d_rgbaImage);
	cudaFree(d_greyImage);

	return 0;
}


