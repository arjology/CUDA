// C++ prgroam to generate Gaussian filter 
#include <cmath> 
#include <iomanip> 
#include <iostream> 
#include "cuda_runtime.h"

#include "opencv2/opencv.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/cuda.hpp"

using namespace cv;
using namespace cv::cuda;

void check(cudaError x) {
    fprintf(stderr, "%s\n", cudaGetErrorString(x));
}

void showMatrix2(double* v1, int width, int height) {
    printf("---------------------\n");
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            printf("%f ", v1[i * width + j]);
        }
        printf("\n");
    }
}

// Function to create Gaussian filter 
__global__
void FilterCreation(double* GKernel, int dim) 
{ 
    // intialising standard deviation to 1.0 
    double sigma = 4.0; 
    double r, s = 2.0 * sigma * sigma; 
    double val;
  
    // sum is for normalization 
    double sum = 0.0; 
  
    // generating dim x dim kernel
    int filterWidth = dim / 2;
    for (int x = -filterWidth; x <= filterWidth; x++) { 
        for (int y = -filterWidth; y <= filterWidth; y++) { 
            const int currentOffset = (x + filterWidth)*dim + (y + filterWidth);
            r = sqrt((float) x * x + y * y); 
            val = (exp(-(r * r) / s)) / (M_PI * s); 
            GKernel[currentOffset] = val;
            sum += GKernel[currentOffset]; 
        } 
    } 
	int kernelLoc;
    // normalising the Kernel 
    for (int i = 0; i < dim; ++i) 
        for (int j = 0; j < dim; ++j) 
            kernelLoc = i*dim + j;
            GKernel[kernelLoc] /= sum; 
}

__global__ 
void kernel(double* tab, double* gaussian, int width, int height, int pitch) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;
    if (row < width && col < height) {
        *( ((double *)(((char *)tab) + (row * pitch))) + col) = 1.0f;
    }
}
  
// Driver program to test above function 
int main() 
{
    size_t pitch;
    int N = 1<<20;
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    printf("Block Size: %d\tNum Blocks: %d", blockSize, numBlocks);
 	getchar();

    int dim = 15;
    double* GKernel;
	double* DKernel;
    int filterSize = dim * dim * sizeof(double);
    check(cudaMallocPitch(&GKernel, &pitch, dim * sizeof(double), dim));
	DKernel = (double*)malloc(filterSize);
    check(cudaMemset(GKernel, 0, filterSize));
    FilterCreation<<<blockSize, numBlocks>>>(GKernel, dim); 
	check( cudaMemcpy2D(DKernel, dim*sizeof(double), GKernel, pitch, dim*sizeof(double), dim, cudaMemcpyDeviceToHost) );

	int loc, j;
    for (int i = 0; i < dim; ++i) { 
        for (j = 0; j < dim; ++j)
			loc = i*dim + j;
			if (j == 0) { std::cout << "\n"; } 
            std::cout << DKernel[loc] << "\t"; 
        std::cout << std::endl; 
    } 
	getchar();
	int imgSize = 16;
    double* d_tab;
	double* h_tab;
    int realSize = imgSize * imgSize * sizeof(double);
    check(cudaMallocPitch(&d_tab, &pitch, imgSize * sizeof(double), imgSize));
    h_tab = (double*)malloc(realSize);
    check( cudaMemset(d_tab, 0, realSize) );
    dim3 grid(4, 4);
    dim3 block(4, 4);
    kernel <<<grid, block>>>(d_tab, GKernel, imgSize, imgSize, pitch);
    check(cudaMemcpy2D(h_tab, imgSize*sizeof(double), d_tab, pitch, imgSize*sizeof(double), imgSize, cudaMemcpyDeviceToHost));
    showMatrix2(h_tab, imgSize, imgSize);
    printf("\nPitch size: %d \n", pitch);
    getchar();
    return 0;
}
