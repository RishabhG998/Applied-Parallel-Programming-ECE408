#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
__constant__ float const_mem_matrix[15000];

__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    const int Height_out = Height - K + 1;// Define output height and width
    const int Width_out = Width - K + 1;
    
    const int Width_grid = (Width_out + TILE_WIDTH - 1) / TILE_WIDTH; // Compute number of thread blocks needed to cover the output tensor

    // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    
#define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0] // Define macros to simplify indexing of tensors
#define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
#define mask_4d(i3, i2, i1, i0) const_mem_matrix[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here

    int H = (blockIdx.y / Width_grid) * TILE_WIDTH + threadIdx.y; // Compute the index of the pixel to be computed by the current thread
    int W = (blockIdx.y % Width_grid) * TILE_WIDTH + threadIdx.x;

    // Check if the thread is within the bounds of the output tensor
    if (H < Height_out && W < Width_out){ // for all height and width pixel values
        float inter = 0.0f; // declaring a temp variable
        for (int C = 0; C < Channel; C++) { // sum over all channels
            for (int k = 0; k < K; k++){ // loop over KxK filter
                for (int i = 0; i < K; i++){
                    // Compute the dot product of the input tensor and filter kernel
                    inter += in_4d(blockIdx.z, C, H + k, W + i) * mask_4d(blockIdx.x, C, k, i); // calculating convolution and adding the intermediate results to inter variable
                }
            }
        }
        out_4d(blockIdx.z, blockIdx.x, H, W) = inter; // storing the final results in out_4d
    }

    #undef out_4d // Undefine macros
    #undef in_4d
    #undef mask_4d
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    int out_size = ((Height_out*Width_out) * Map_out * Batch) * sizeof(float); // output size is batchsize * output channels * size of each output image
    int in_size = (Height*Width) * Channel * Batch * sizeof(float); // input size is input image dimensions * channels * batchsize
    int k_size = (K*K) * Map_out * Channel * sizeof(float); //each filter times input channels and output feature maps

    cudaMalloc((void**)device_input_ptr, in_size);
    // cudaMalloc((void**)device_mask_ptr, k_size);
    cudaMalloc((void**)device_output_ptr, out_size);
    

    cudaMemcpy(*device_input_ptr, host_input, in_size, cudaMemcpyHostToDevice);
    // cudaMemcpy(*device_mask_ptr, host_mask, k_size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(const_mem_matrix, host_mask, k_size);

}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel
    const int Height_ = ((Height - K + 1) + TILE_WIDTH - 1) / TILE_WIDTH; // Calculate the dimensions of the kernel block and grid
    const int Width_ = ((Width - K + 1) + TILE_WIDTH - 1) / TILE_WIDTH;

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1); // Set the dimensions of the kernel block and grid
    dim3 gridDim(Map_out, Width_ * Height_ , Batch);

    conv_forward_kernel<<<gridDim, blockDim>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);

}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    int out_size = (Height_out*Width_out) * Map_out * Batch * sizeof(float);

    // Copy the output back to host

    cudaMemcpy(host_output, device_output, out_size, cudaMemcpyDeviceToHost);

    // Free device memory

    cudaFree(device_output);
    cudaFree(device_input);
    // cudaFree(device_mask);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
