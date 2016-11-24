
#include <iostream>
#include "timer.h"
#include "utils.h"
#include <string>
#include <stdio.h>
#include "reference_calc.h"
#include "compare.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>

void your_rgba_to_greyscale(const uchar4 * const h_rgbaImage, 
		uchar4 * const d_rgbaImage,
		unsigned char* const d_greyImage,
		size_t numRows, size_t numCols);

//include the definitions of the above functions

cv::Mat imageRGBA;
cv::Mat imageGrey;

uchar4        *d_rgbaImage__;
unsigned char *d_greyImage__;

size_t numRows() { return imageRGBA.rows; }
size_t numCols() { return imageRGBA.cols; }

void preProcess(uchar4 **inputImage, unsigned char **greyImage,
		uchar4 **d_rgbaImage, unsigned char **d_greyImage,
		const std::string &filename) {
	//make sure the context initializes ok
	checkCudaErrors(cudaFree(0));

	cv::Mat image;
	image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
	if (image.empty()) {
		std::cerr << "Couldn't open file: " << filename << std::endl;
		exit(1);
	}

	cv::cvtColor(image, imageRGBA, CV_BGR2RGBA);

	//allocate memory for the output
	imageGrey.create(image.rows, image.cols, CV_8UC1);

	*inputImage = (uchar4 *)imageRGBA.ptr<unsigned char>(0);
	*greyImage  = imageGrey.ptr<unsigned char>(0);

	const size_t numPixels = numRows() * numCols();
	//allocate memory on the device for both input and output
	checkCudaErrors(cudaMalloc(d_rgbaImage, sizeof(uchar4) * numPixels));
	checkCudaErrors(cudaMalloc(d_greyImage, sizeof(unsigned char) * numPixels));
	checkCudaErrors(cudaMemset(*d_greyImage, 0, numPixels * sizeof(unsigned char))); //make sure no memory is left laying around

	//copy input array to the GPU
	checkCudaErrors(cudaMemcpy(*d_rgbaImage, *inputImage, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice));

	d_rgbaImage__ = *d_rgbaImage;
	d_greyImage__ = *d_greyImage;
	cv::imshow("display",image);
	cv::waitKey(0);
}

void postProcess(const std::string& output_file, unsigned char* data_ptr) {
	cv::Mat output(numRows(), numCols(), CV_8UC1, (void*)data_ptr);
	//output the image
	cv::imwrite(output_file.c_str(), output);
	cv::imshow("display",output);
	cv::waitKey(0);
}

void cleanup()
{
	//cleanup
	cudaFree(d_rgbaImage__);
	cudaFree(d_greyImage__);
}

void generateReferenceImage(std::string input_filename, std::string output_filename)
{
	cv::Mat reference = cv::imread(input_filename, CV_LOAD_IMAGE_GRAYSCALE);

	cv::imwrite(output_filename, reference);

}

/*
 *
 */

int main(int argc, char **argv) {
	uchar4        *h_rgbaImage, *d_rgbaImage;
	unsigned char *h_greyImage, *d_greyImage;

	std::string input_file;
	std::string output_file;
	std::string reference_file;
	double perPixelError = 0.0;
	double globalError   = 0.0;
	bool useEpsCheck = false;
	if(argc > 1)
	{
		switch (argc)
		{
		case 2:
			input_file = std::string(argv[1]);
			output_file = "output.png";
			reference_file = "reference.png";
			break;
		default:
			std::cerr << "Usage: input_file [output_filename] [reference_filename] [perPixelError] [globalError]" << std::endl;
			exit(1);
		}
	}
	else
	{
		std::cout << "Input the image name: " << std::endl;
		std::cin >> input_file;
		output_file = "output.png";
		reference_file = "reference.png";
	}


	//load the image and give us our input and output pointers
	preProcess(&h_rgbaImage, &h_greyImage, &d_rgbaImage, &d_greyImage, input_file);

	GpuTimer timer;
	timer.Start();
	//call the students' code
	your_rgba_to_greyscale(h_rgbaImage, d_rgbaImage, d_greyImage, numRows(), numCols());
	timer.Stop();
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	int err = printf("Your code ran in: %f msecs.\n", timer.Elapsed());

	if (err < 0) {
		//Couldn't print! Probably the student closed stdout - bad news
		std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
		exit(1);
	}

	size_t numPixels = numRows()*numCols();
	checkCudaErrors(cudaMemcpy(h_greyImage, d_greyImage, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost));

	//check results and output the grey image
	postProcess(output_file, h_greyImage);

	referenceCalculation(h_rgbaImage, h_greyImage, numRows(), numCols());

	postProcess(reference_file, h_greyImage);

	//generateReferenceImage(input_file, reference_file);
	compareImages(reference_file, output_file, useEpsCheck, perPixelError,
			globalError);

	cleanup();

	return 0;
}
