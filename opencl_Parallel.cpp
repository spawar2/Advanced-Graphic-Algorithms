void processUsingCuda(std::string input_file, std::string output_file) {
	// pointers to images in CPU's memory (h_) and GPU's memory (d_)
	uchar4        *h_rgbaImage, *d_rgbaImage;
	unsigned char *h_greyImage, *d_greyImage;
 
	//load the image and give us our input and output pointers
	preProcess(&h_rgbaImage, &h_greyImage, &d_rgbaImage, &d_greyImage, input_file);
 
	GpuTimer timer;
	timer.Start();
	// here is where the conversion actually happens
	rgbaToGreyscaleCuda(h_rgbaImage, d_rgbaImage, d_greyImage, numRows(), numCols());
	timer.Stop();
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
 
	int err = printf("Implemented CUDA code ran in: %f msecs.\n", timer.Elapsed());
 
	if (err < 0) {
		//Couldn't print!
		std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
		exit(1);
	}
 
	size_t numPixels = numRows()*numCols();
	checkCudaErrors(cudaMemcpy(h_greyImage, d_greyImage, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost));
 
	//check results and output the grey image
	postProcess(output_file, h_greyImage);
}




#ifndef GRAYSCALE_H__
#define GRAYSCALE_H__
 
#include <stdio.h>
#include "utils.h"
 
//  kernel which is run in parallel by many GPU threads.
__global__
void rgbaToGreyscaleCudaKernel(const uchar4* const rgbaImage,
		unsigned char* const greyImage,
		const int numRows, const int numCols)
{
	//First create a mapping from the 2D block and grid locations
	//to an absolute 2D location in the image, then use that to
	//calculate a 1D offset
	const long pointIndex = threadIdx.x + blockDim.x*blockIdx.x;
 
	if(pointIndex<numRows*numCols) { // this is necessary only if too many threads are started
		uchar4 const imagePoint = rgbaImage[pointIndex];
		greyImage[pointIndex] = .299f*imagePoint.x + .587f*imagePoint.y  + .114f*imagePoint.z;
	}
}
 
// Parallel implementation for running on GPU using multiple threads.
void rgbaToGreyscaleCuda(const uchar4 * const h_rgbaImage, uchar4 * const d_rgbaImage,
		unsigned char* const d_greyImage, const size_t numRows, const size_t numCols)
{
	const int blockThreadSize = 256;
	const int numberOfBlocks = 1 + ((numRows*numCols - 1) / blockThreadSize); // a/b rounded up
	const dim3 blockSize(blockThreadSize, 1, 1);
	const dim3 gridSize(numberOfBlocks , 1, 1);
	rgbaToGreyscaleCudaKernel<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);
}




