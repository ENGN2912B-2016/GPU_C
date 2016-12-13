#include <iostream>
#include <stdio.h>
#include <iomanip>
#include <cuda.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <helper_cuda.h>

__global__
void separateChannel(const uchar4* const inputImageRGBA, const size_t numRows, const size_t numCols,
		uchar* const redChannel, uchar* const greenChannel, uchar* const blueChannel)
{
	int m = blockIdx.x*blockDim.x+threadIdx.x;
	int n = blockIdx.y*blockDim.y+threadIdx.y;
	if (m >= numCols || n >= numRows)
	{
		return;
	}
	uchar4 rgba = inputImageRGBA[n*numCols + m];
	redChannel[n*numCols + m] = rgba.x;
	greenChannel[n*numCols + m] = rgba.y;
	blueChannel[n*numCols + m] = rgba.z;
}

__global__
void gaussianBlur(const uchar* const inputChannel, uchar* const outputChannel,
		const size_t numRows, const size_t numCols, const float* const filter, const int filterWidth)
{
	//assert(filterWidth % 2 == 1);
	int m = blockIdx.x*blockDim.x+threadIdx.x;
	int n = blockIdx.y*blockDim.y+threadIdx.y;
	float result = 0.f;
	if (m >= numCols ||	n >= numRows)
	{
		return;
	}
	//For every value in the filter around the pixel (c, r)
	for (int filter_r = -filterWidth / 2; filter_r <= filterWidth / 2; ++filter_r) {
		for (int filter_c = -filterWidth / 2; filter_c <= filterWidth / 2; ++filter_c) {
			//Find the global image position for this filter position
			int image_r = min(max(n + filter_r, 0), (int)(numRows - 1));
			int image_c = min(max(m + filter_c, 0), (int)(numCols - 1));

			float image_value = (float)(inputChannel[image_r * numCols + image_c]);
			int filter_pos=(filter_r + filterWidth / 2) * filterWidth + filter_c + filterWidth / 2;
			float filter_value = filter[filter_pos];

			result += image_value * filter_value;
		}
	}

	outputChannel[n * numCols + m] = result;
}

__global__
void recombineChannel(const uchar* const redChannel, const uchar* const greenChannel,
		const uchar* const blueChannel, uchar4* const outputImageRGBA, const size_t numRows, const size_t numCols)
{
	//	  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
	//	                                        blockIdx.y * blockDim.y + threadIdx.y);
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	const int thread_1D_pos = y * numCols + x;

	if (x >= numCols || y >= numRows)
		return;

	uchar red   = redChannel[thread_1D_pos];
	uchar green = greenChannel[thread_1D_pos];
	uchar blue  = blueChannel[thread_1D_pos];

	//Alpha should be 255
	uchar4 outputPixel = make_uchar4(red, green, blue, 255);

	outputImageRGBA[thread_1D_pos] = outputPixel;
}

void doBlurImage(uchar4* const d_inputImageRGBA,
		uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols, const size_t numPixels,
		uchar* d_redBlurred, uchar* d_greenBlurred, uchar* d_blueBlurred, float* d_filter)
{
	uchar *d_red, *d_green, *d_blue;
	int filterWidth = 9;
	//	float *d_filter;
	//allocate memory for origin channel
	checkCudaErrors(cudaMalloc(&d_red,   sizeof(uchar) * numPixels));
	checkCudaErrors(cudaMalloc(&d_green, sizeof(uchar) * numPixels));
	checkCudaErrors(cudaMalloc(&d_blue,  sizeof(uchar) * numPixels));

	const dim3 blockSize(32, 32, 1);
	const dim3 gridSize(numCols/32+1, numRows/32+1, 1);
	separateChannel<<<gridSize, blockSize >>>(d_inputImageRGBA, numRows, numCols, d_red, d_green, d_blue);
	cudaDeviceSynchronize();checkCudaErrors(cudaGetLastError());

	gaussianBlur<<<gridSize, blockSize >>>(d_red, d_redBlurred, numRows, numCols, d_filter, filterWidth);
	gaussianBlur<<<gridSize, blockSize >>>(d_green, d_greenBlurred, numRows, numCols, d_filter, filterWidth);
	gaussianBlur<<<gridSize, blockSize >>>(d_blue, d_blueBlurred, numRows, numCols, d_filter, filterWidth);
	cudaDeviceSynchronize();checkCudaErrors(cudaGetLastError());

	recombineChannel<<<gridSize, blockSize>>>(d_redBlurred, d_greenBlurred, d_blueBlurred,
			d_outputImageRGBA, numRows, numCols);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	checkCudaErrors(cudaFree(d_red));
	checkCudaErrors(cudaFree(d_green));
	checkCudaErrors(cudaFree(d_blue));
}
