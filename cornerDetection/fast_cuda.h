
#ifndef FAST_CUDA_H_
#define FAST_CUDA_H_

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/opencv.hpp>
#include "corner.h"

__device__
int position(int m,int n,int width);

__global__
void fast(uchar* image, int width, int height,Corner* d_corner,int gridsize_x, int gridsize_y, const int threshold);

__global__
void nms(uchar* image, Corner* d_corner, int width, int height);

extern void fast_corner_9(const cv::Mat& image,std::vector<cv::KeyPoint>& keypoint,const int threshold);



#endif /* FAST_CUDA_H_ */
