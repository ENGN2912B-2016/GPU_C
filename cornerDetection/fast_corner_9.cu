#include <iostream>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include <timer.h>
#include <string>
#include <vector>
#include "corner.h"
#include "fast_cuda.h"

void fast_corner_9(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoint,const int threshold)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	uchar* d_data;                // create a pointer
	size_t imSize = image.cols*image.rows;
	Corner* h_corner = new Corner[imSize];
	Corner* d_corner;
	checkCudaErrors(cudaMalloc((void**) &d_corner,sizeof(Corner) * imSize));
	checkCudaErrors(cudaMalloc((void**) &d_data, sizeof(uchar) * imSize)); // create memory on the gpu and pass a pointer to the host
	checkCudaErrors(cudaMemcpy(d_data, image.data, sizeof(uchar) * imSize, cudaMemcpyHostToDevice));// copy from the image data to the gpu memory you reserved
	dim3 blocksize(16, 16);
	dim3 gridsize((image.cols-1)/blocksize.x+1, (image.rows-1)/blocksize.y+1, 1);
	cudaEventRecord(start);
	fast<<<gridsize,blocksize>>>(d_data, image.cols, image.rows,d_corner,gridsize.x,gridsize.y,threshold); // processed data on the gpu
	cudaEventRecord(stop);
	nms<<<gridsize,blocksize>>>(d_data, d_corner, image.cols, image.rows);
	checkCudaErrors(cudaMemcpy(h_corner, d_corner, sizeof(Corner) * imSize, cudaMemcpyDeviceToHost));
	for(int i = 0;i < imSize; i++)
	{
		cv::KeyPoint temp;
		if(h_corner[i].set == 1)
		{
		    temp=cv::KeyPoint(i%image.cols, i/image.cols ,-1,-1,0,0,-1);
			keypoint.push_back(temp);
		}
	}
	float elptime;
	cudaEventElapsedTime(&elptime,start,stop);
	std::cout << "Number of corners: "<< keypoint.size() << std::endl;
	std::cout << "Detecting time: "<< elptime << "ms" << std::endl;
	delete[] h_corner;
	cudaFree(d_corner);
	cudaFree(d_data);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

}
