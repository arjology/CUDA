// C++ prgroam to generate Gaussian filter 
#include <cmath> 
#include <iomanip> 
#include <iostream> 
#include "cuda_runtime.h"

#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include <opencv2/cudaarithm.hpp>

void check(cudaError x) {
    fprintf(stderr, "%s\n", cudaGetErrorString(x));
}

void showMatrix2(double* v1, int width, int height) {
    printf("---------------------\n");
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            printf("%8.6lf ", v1[i * width + j]);
        }
        printf("\n");
    }
}

// Function to create Gaussian filter 
__global__
void FilterCreation(double* GKernel, int dim) 
{ 
    // intialising standard deviation to 1.0 
    double sigma = 10.0; 
    double r, s = 2.0 * sigma * sigma; 
  
    // sum is for normalization 
    double sum = 0.0; 
  
    // generating dim x dim kernel
    int filterWidth = dim / 2;
    for (int x = -filterWidth; x <= filterWidth; x++) { 
        for (int y = -filterWidth; y <= filterWidth; y++) { 
            r = sqrt((float) x * x + y * y); 
            
            GKernel[(x + filterWidth)*dim + (y + filterWidth)] = (exp(-(r * r) / s)) / (M_PI * s);
            sum += GKernel[(x + filterWidth)*dim + (y + filterWidth)];
        } 
    }
    // normalising the Kernel 
    for (int i = 0; i < dim; ++i) 
        for (int j = 0; j < dim; ++j) 
            GKernel[i*dim + j] /= sum; 
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
    cv::cuda::DeviceInfo info = cv::cuda::getDevice();
    std::cout << info.name() << std::endl;

    size_t pitch;
    int N = 1<<20;
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    printf("Block Size: %d\tNum Blocks: %d", blockSize, numBlocks);
 	getchar();

    // Create Gaussian circle
    int dim = 15;
    double* GKernel;
	double* DKernel;
    int filterSize = dim * dim * sizeof(double);
    check(cudaMallocPitch(&GKernel, &pitch, dim * sizeof(double), dim));
	DKernel = (double*)malloc(filterSize);
    check(cudaMemset(GKernel, 0, filterSize));
    FilterCreation<<<blockSize, numBlocks>>>(GKernel, dim); 
	check(cudaMemcpy2D(DKernel, dim*sizeof(double), GKernel, pitch, dim*sizeof(double), dim, cudaMemcpyDeviceToHost));
    showMatrix2(DKernel, dim, dim);
	getchar();

	// int imgSize = 16;
    // double* d_tab;
	// double* h_tab;
    // int realSize = imgSize * imgSize * sizeof(double);
    // check(cudaMallocPitch(&d_tab, &pitch, imgSize * sizeof(double), imgSize));
    // h_tab = (double*)malloc(realSize);
    // check( cudaMemset(d_tab, 0, realSize) );
    // kernel <<<grid, block>>>(d_tab, GKernel, imgSize, imgSize, pitch);
    // check(cudaMemcpy2D(h_tab, imgSize*sizeof(double), d_tab, pitch, imgSize*sizeof(double), imgSize, cudaMemcpyDeviceToHost));
    // showMatrix2(h_tab, imgSize, imgSize);
    // printf("\nPitch size: %d \n", pitch);
    // getchar();

    // // Read blank image and display
    // uint8_t *imgPtr;
    // cv::Mat srcImg, dstImg;
    // cv::cuda::GpuMat gpuImg;

    // srcImg = cv::imread("figs/blank.jpg", cv::IMREAD_GRAYSCALE);
    // gpuImg.upload(srcImg);
    // cudaMalloc((void **)&imgPtr, gpuImg.rows*gpuImg.step);
    // check(cudaMemcpyAsync(imgPtr, gpuImg.ptr<uint8_t>(), gpuImg.rows*gpuImg.step, cudaMemcpyDeviceToDevice));
    // cv::cuda::GpuMat gpuSrc(srcImg.rows, srcImg.cols, srcImg.type(), imgPtr, gpuImg.step);
    // gpuImg.download(dstImg);
    // // cv::imshow("test", dstImg);
    // // cv::waitKey(0);
    // cv::imwrite("figs/blank_mod.png", dstImg);

    return 0;
}
