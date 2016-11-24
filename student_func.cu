
#include "utils.h"

__global__
void rgba_to_greyscale(const uchar4* const rgbaImage,
                       unsigned char* const greyImage,
                       int numRows, int numCols)
{

	//First create a mapping from the 2D block and grid locations
	//to an absolute 2D location in the image, then use that to
	//calculate a 1D offset
	//  int n = threadIdx.x;
	//  int m = blockIdx.x;
	//  uchar4 rgba=rgbaImage[numCols*m+n];
	//  greyImage[numCols*m+n]=.299f * rgba.x + .587f * rgba.y + .114f * rgba.z;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(x <= numCols && y <= numRows)
	{
		uchar4 rgba=rgbaImage[numCols*y+x];
		greyImage[numCols*y+x]=.299f * rgba.x + .587f * rgba.y + .114f * rgba.z;
	}
}

void your_rgba_to_greyscale(const uchar4 * const h_rgbaImage, uchar4 * const d_rgbaImage,
                            unsigned char* const d_greyImage, size_t numRows, size_t numCols)
{
  //You must fill in the correct sizes for the blockSize and gridSize
  //currently only one block with one thread is being launched
  const dim3 blockSize(32, 32, 1);  //TODO
  const dim3 gridSize(numCols/32+1, numRows/32+1, 1);  //TODO
  rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);
  
  cudaDeviceSynchronize(); 
  checkCudaErrors(cudaGetLastError());

}
