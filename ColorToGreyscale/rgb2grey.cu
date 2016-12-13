

#include <iostream>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <opencv2/opencv.hpp>

__global__
void doConversion(const uchar4* const rgbaImage, uchar* const greyImage, int numRows, int numCols)
{
	int m = threadIdx.x;
	int n = blockIdx.x;
	uchar4 rgbValue = rgbaImage[n * numCols + m];
	greyImage[n * numCols + m] = 0.299f * rgbValue.x + 0.587f * rgbValue.y + 0.114f *  rgbValue.z;
}




void rgb2grey(const uchar4* const rgbaImage, uchar* const greyImage, size_t numRows, size_t numCols)
{
	const dim3 blockSize(numCols, 1, 1);
	const dim3 gridSize(numRows, 1, 1);
	doConversion<<<gridSize, blockSize>>>(rgbaImage, greyImage, numRows, numCols);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
}
