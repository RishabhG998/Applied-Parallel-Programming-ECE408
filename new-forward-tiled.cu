#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define Tile_Width 16

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

    const int Height_out = Height - K + 1; // Compute the output height and width dimensions
    const int Width_out = Width - K + 1;
    const int Width_grid = (Width_out + Tile_Width - 1) / Tile_Width; // Compute the number of blocks required for the output width dimension
    int Blk_width = Tile_Width + K - 1; // Compute the block width to accommodate the kernel

    extern __shared__ float Shared_Mem[]; // Allocate shared memory
    float* Shared_Mem_obj = &Shared_Mem[0]; // Create a pointer to the start of the shared memory block

#define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
#define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
#define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    int bidz = blockIdx.z;
    int bidx = blockIdx.x;
    int bidy = blockIdx.y;
    int t = threadIdx.x + threadIdx.y * Blk_width;
    int height = (bidy / Width_grid) * Tile_Width + threadIdx.y; // Compute the input height and width indices
    int width = (bidy % Width_grid) * Tile_Width + threadIdx.x;
    float inter = 0.0f;
    
    for(int c = 0; c < Channel; c++){
        // copy data from global memory to shared memory
        if((height < Height) && (width < Width))
            Shared_Mem_obj[t] = in_4d(bidz, c, height, width); // Copy data from global memory to shared memory
        else
            Shared_Mem_obj[t] = 0.0f;
        __syncthreads();

        // convolution
        if((height < Height_out) && (width < Width_out)){
            for(int p = 0; p < K; p++)
                for(int q = 0; q < K; q++)
                    inter += in_4d(bidz, c, height + p, width + q) * mask_4d(bidx, c, p, q);
        }
        __syncthreads();
    }
    if((height < Height_out) && (width < Width_out))
        out_4d(bidz, bidx, height, width) = inter;

#undef out_4d
#undef in_4d
#undef mask_4d
}


__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU
    const int Height_out = Height - K + 1; // Calculate the output dimensions
    const int Width_out = Width - K + 1;
    int value =  Map_out * Channel * K * K * sizeof(float); // Calculate the sizes of the relevant data structures
    int value_2 = Batch * Channel * Height * Width * sizeof(float);
    int value_3 = Batch * Map_out * Height_out * Width_out * sizeof(float);

    cudaMalloc((void**)device_output_ptr, value_3); // Allocate memory on the GPU for the output, input, and mask
    cudaMalloc((void**)device_input_ptr, value_2);
    cudaMalloc((void**)device_mask_ptr,value);

    cudaMemcpy(*device_input_ptr, host_input, value_2, cudaMemcpyHostToDevice); // Copy the input and mask data from the host to the device
    cudaMemcpy(*device_mask_ptr, host_mask, value, cudaMemcpyHostToDevice);

}


__host__ void GPUInterface::conv_forward_gpu(float *device_y, const float *device_x, const float *device_k, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel
    const int Height_ = ((Height - K + 1) + Tile_Width - 1) / Tile_Width; // Calculate the dimensions of the kernel block and grid
    const int Width_ = ((Width - K + 1) + Tile_Width - 1) / Tile_Width;
    int Blk_width = Tile_Width + K - 1;

    dim3 blockDim(Blk_width, Blk_width, 1); // Set the dimensions of the kernel block and grid
    dim3 gridDim(Map_out, Width_ * Height_ , Batch);

    size_t Shared_Mem_obj1 = (Blk_width) * (Blk_width) * sizeof(float); // Allocate shared memory for the kernel

    conv_forward_kernel<<<gridDim, blockDim, Shared_Mem_obj1>>>(device_y, device_x, device_k, Batch, Map_out, Channel, Height, Width, K);
    cudaDeviceSynchronize();
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_y, float *device_x, float *device_k, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    cudaMemcpy(host_output, device_y, Batch * Map_out * Height_out * Width_out * sizeof(float), cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(device_y);
    cudaFree(device_x);
    cudaFree(device_k);
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