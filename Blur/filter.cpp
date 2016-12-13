
#include <iostream>
#include <math.h>


void constructFilter(float* const h_filter, const int blurKernelWidth)
{
	float filterSum = 0.f; //for normalization
	const float blurKernelSigma = 2;

	for (int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r) {
		for (int c = -blurKernelWidth/2; c <= blurKernelWidth/2; ++c) {
			float filterValue = expf( -(float)(c * c + r * r) / (2.f * blurKernelSigma * blurKernelSigma));
			h_filter[(r + blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2] = filterValue;
			filterSum += filterValue;
		}
	}

	float normalizationFactor = 1.f / filterSum;

	for (int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r) {
		for (int c = -blurKernelWidth/2; c <= blurKernelWidth/2; ++c) {
			h_filter[(r + blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2] *= normalizationFactor;
		}
	}
}
