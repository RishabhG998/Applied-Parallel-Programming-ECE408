#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 8

__global__ void conv_forward_kernel(float* __restrict__ output, const float* __restrict__ input, const float* __restrict__ mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
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

    const int Height_out = Height - K + 1; // Calculate the width and height of each output
    const int Width_out = Width - K + 1;
    

    // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    float var = 0.0f;

    // Insert your GPU convolution kernel code here
    const int Width_grid = (Width_out + TILE_WIDTH - 1) / TILE_WIDTH;
    int height = (blockIdx.y / Width_grid) * TILE_WIDTH + threadIdx.y; // Calculate the height and width indices for the current thread
    int width = (blockIdx.y % Width_grid) * TILE_WIDTH + threadIdx.x;
    
    if((height < Height_out) && (width < Width_out)){
        for(int c = 0; c < Channel; c++){
            var = var + in_4d(blockIdx.z,c,height+0,width+0) * mask_4d(blockIdx.x,c,0,0) + in_4d(blockIdx.z,c,height+0,width+1) * mask_4d(blockIdx.x,c,0,1) + in_4d(blockIdx.z,c,height+0,width+2) * mask_4d(blockIdx.x,c,0,2) + in_4d(blockIdx.z,c,height+0,width+3) * mask_4d(blockIdx.x,c,0,3)+ in_4d(blockIdx.z,c,height+0,width+4) * mask_4d(blockIdx.x,c,0,4)+ in_4d(blockIdx.z,c,height+0,width+5) * mask_4d(blockIdx.x,c,0,5)+ in_4d(blockIdx.z,c,height+0,width+6) * mask_4d(blockIdx.x,c,0,6) ;
            
            var = var + in_4d(blockIdx.z,c,height+1,width+0) * mask_4d(blockIdx.x,c,1,0) + in_4d(blockIdx.z,c,height+1,width+1) * mask_4d(blockIdx.x,c,1,1) + in_4d(blockIdx.z,c,height+1,width+2) * mask_4d(blockIdx.x,c,1,2) + in_4d(blockIdx.z,c,height+1,width+3) * mask_4d(blockIdx.x,c,1,3)+ in_4d(blockIdx.z,c,height+1,width+4) * mask_4d(blockIdx.x,c,1,4)+ in_4d(blockIdx.z,c,height+1,width+5) * mask_4d(blockIdx.x,c,1,5)+ in_4d(blockIdx.z,c,height+1,width+6) * mask_4d(blockIdx.x,c,1,6) ;

            var = var + in_4d(blockIdx.z,c,height+2,width+0) * mask_4d(blockIdx.x,c,2,0) + in_4d(blockIdx.z,c,height+2,width+1) * mask_4d(blockIdx.x,c,2,1) + in_4d(blockIdx.z,c,height+2,width+2) * mask_4d(blockIdx.x,c,2,2) + in_4d(blockIdx.z,c,height+2,width+3) * mask_4d(blockIdx.x,c,2,3)+ in_4d(blockIdx.z,c,height+2,width+4) * mask_4d(blockIdx.x,c,2,4)+ in_4d(blockIdx.z,c,height+2,width+5) * mask_4d(blockIdx.x,c,2,5)+ in_4d(blockIdx.z,c,height+2,width+6) * mask_4d(blockIdx.x,c,2,6) ;

            var = var + in_4d(blockIdx.z,c,height+3,width+0) * mask_4d(blockIdx.x,c,3,0) + in_4d(blockIdx.z,c,height+3,width+1) * mask_4d(blockIdx.x,c,3,1) + in_4d(blockIdx.z,c,height+3,width+2) * mask_4d(blockIdx.x,c,3,2) + in_4d(blockIdx.z,c,height+3,width+3) * mask_4d(blockIdx.x,c,3,3)+ in_4d(blockIdx.z,c,height+3,width+4) * mask_4d(blockIdx.x,c,3,4)+ in_4d(blockIdx.z,c,height+3,width+5) * mask_4d(blockIdx.x,c,3,5)+ in_4d(blockIdx.z,c,height+3,width+6) * mask_4d(blockIdx.x,c,3,6) ;

            var = var + in_4d(blockIdx.z,c,height+4,width+0) * mask_4d(blockIdx.x,c,4,0) + in_4d(blockIdx.z,c,height+4,width+1) * mask_4d(blockIdx.x,c,4,1) + in_4d(blockIdx.z,c,height+4,width+2) * mask_4d(blockIdx.x,c,4,2) + in_4d(blockIdx.z,c,height+4,width+3) * mask_4d(blockIdx.x,c,4,3)+ in_4d(blockIdx.z,c,height+4,width+4) * mask_4d(blockIdx.x,c,4,4)+ in_4d(blockIdx.z,c,height+4,width+5) * mask_4d(blockIdx.x,c,4,5)+ in_4d(blockIdx.z,c,height+4,width+6) * mask_4d(blockIdx.x,c,4,6); 

            var = var + in_4d(blockIdx.z,c,height+5,width+0) * mask_4d(blockIdx.x,c,5,0) + in_4d(blockIdx.z,c,height+5,width+1) * mask_4d(blockIdx.x,c,5,1) + in_4d(blockIdx.z,c,height+5,width+2) * mask_4d(blockIdx.x,c,5,2) + in_4d(blockIdx.z,c,height+5,width+3) * mask_4d(blockIdx.x,c,5,3)+ in_4d(blockIdx.z,c,height+5,width+4) * mask_4d(blockIdx.x,c,5,4)+ in_4d(blockIdx.z,c,height+5,width+5) * mask_4d(blockIdx.x,c,5,5)+ in_4d(blockIdx.z,c,height+5,width+6) * mask_4d(blockIdx.x,c,5,6) ;

            var = var + in_4d(blockIdx.z,c,height+6,width+0) * mask_4d(blockIdx.x,c,6,0) + in_4d(blockIdx.z,c,height+6,width+1) * mask_4d(blockIdx.x,c,6,1) + in_4d(blockIdx.z,c,height+6,width+2) * mask_4d(blockIdx.x,c,6,2) + in_4d(blockIdx.z,c,height+6,width+3) * mask_4d(blockIdx.x,c,6,3)+ in_4d(blockIdx.z,c,height+6,width+4) * mask_4d(blockIdx.x,c,6,4)+ in_4d(blockIdx.z,c,height+6,width+5) * mask_4d(blockIdx.x,c,6,5)+ in_4d(blockIdx.z,c,height+6,width+6) * mask_4d(blockIdx.x,c,6,6) ;

            var = var + in_4d(blockIdx.z,c,height+7,width+0) * mask_4d(blockIdx.x,c,7,0) + in_4d(blockIdx.z,c,height+7,width+1) * mask_4d(blockIdx.x,c,7,1) + in_4d(blockIdx.z,c,height+7,width+2) * mask_4d(blockIdx.x,c,7,2) + in_4d(blockIdx.z,c,height+7,width+3) * mask_4d(blockIdx.x,c,7,3)+ in_4d(blockIdx.z,c,height+7,width+4) * mask_4d(blockIdx.x,c,7,4)+ in_4d(blockIdx.z,c,height+7,width+5) * mask_4d(blockIdx.x,c,7,5)+ in_4d(blockIdx.z,c,height+7,width+6) * mask_4d(blockIdx.x,c,7,6); 

            
        }
        out_4d(blockIdx.z, blockIdx.x, height, width) = var; // storing the final results in out_4d
    }


    #undef out_4d
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
    cudaMalloc((void**)device_mask_ptr, k_size);
    cudaMalloc((void**)device_output_ptr, out_size);

    cudaMemcpy(*device_input_ptr, host_input, in_size, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, k_size, cudaMemcpyHostToDevice);

}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    dim3 dimGrid(Map_out, ((Width_out + TILE_WIDTH -1)/TILE_WIDTH) * ((Height_out + TILE_WIDTH -1)/TILE_WIDTH), Batch);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    conv_forward_kernel<<<dimGrid,dimBlock>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);

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
    cudaFree(device_mask);
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
