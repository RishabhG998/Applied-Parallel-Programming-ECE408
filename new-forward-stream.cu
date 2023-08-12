#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 8

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

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    int W_size = (Width_out + TILE_WIDTH - 1) / TILE_WIDTH;
#define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
#define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
#define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    
    // int H_size = Height_out/TILE_WIDTH;
    int y = blockIdx.y; 
    int H = (y / W_size) * TILE_WIDTH + threadIdx.y;
    int W = (y % W_size) * TILE_WIDTH + threadIdx.x;
    
    if (H < Height_out && W < Width_out){ // for all batches, for all height and width pixel values
        float inter = 0.0f; // declaring a temp variable
        for (int C = 0; C < Channel; C++) { // sum over all channels
            for (int k = 0; k < K; k++){ // loop over KxK filter
                for (int i = 0; i < K; i++){
                    inter += in_4d(blockIdx.z, C, H + k, W + i) * mask_4d(blockIdx.x, C, k, i); // calculating convolution and adding the intermediate results to inter variable
                }
            }
        }
        out_4d(blockIdx.z, blockIdx.x, H, W) = inter; // storing the final results in out_4d
    }

#undef out_4d
#undef in_4d
#undef mask_4d
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU

#define stream_size 10

    // Allocate memory and copy over the relevant data structures to the GPU

    const int Height_out = Height - K + 1; // Calculate output height and width based on input height, width, and filter size.
    const int Weight_out = Width - K + 1;

    int W_size = (Weight_out + TILE_WIDTH - 1) / TILE_WIDTH; // Determine number of thread blocks needed for each dimension based on output height, width, and tile size.
    int H_size = (Height_out + TILE_WIDTH - 1) / TILE_WIDTH;

    float* host_output_temp = (float*)host_output; // Cast the host_output pointer to a float pointer for convenience.

    int input_batch_size = (Batch * Channel * Height * Width) / stream_size; // Calculate the input and output batch sizes per stream.
    int output_batch_size = (Batch * Map_out * Height_out * Weight_out) / stream_size;    

    dim3 gridDim(Map_out, W_size * H_size, Batch/stream_size); // Set the dimensions of the CUDA kernel grid and thread blocks.
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    
    cudaStream_t A[stream_size]; // Create an array of CUDA streams.
    for (int x = 0; x < stream_size; x++){
        cudaStreamCreate(&A[x]); // Create a CUDA stream for each index in the array.
    }

    
    int out_size = ((Height_out*Weight_out) * Map_out * Batch) * sizeof(float); // output size is batchsize * output channels * size of each output image
    int in_size = (Height*Width) * Channel * Batch * sizeof(float); // input size is input image dimensions * channels * batchsize
    int k_size = (K*K) * Map_out * Channel * sizeof(float); //each filter times input channels and output feature maps

    cudaMalloc((void**)device_input_ptr, in_size); // Allocate memory on the device for input, mask, and output data.
    cudaMalloc((void**)device_mask_ptr, k_size);
    cudaMalloc((void**)device_output_ptr, out_size);

    cudaMemcpyAsync(*device_mask_ptr, host_mask, k_size, cudaMemcpyHostToDevice, A[0]); // Copy the filter mask from the host to the device asynchronously using stream 0.

    for (int i = 0; i < stream_size; i++){ // Loop over each stream to perform convolution on a batch subset of the input data.
        int input_offset = input_batch_size * i; // Calculate the input and output batch offsets for the current stream.
        int output_offset = output_batch_size * i;
        // Copy the input batch data from the host to the device asynchronously for the current stream.
        cudaMemcpyAsync((*device_input_ptr) + input_offset, host_input + input_offset, input_batch_size * sizeof(float), cudaMemcpyHostToDevice, A[i]);
        // Launch the convolution forward kernel for the current stream
        conv_forward_kernel<<<gridDim, blockDim, 0, A[i]>>>((*device_output_ptr) + output_offset, (*device_input_ptr) + input_offset, *device_mask_ptr, Batch, Map_out,Channel, Height, Width, K);
        // Copy the output batch data from the device to the host asynchronously for the current stream.
        cudaMemcpyAsync(host_output_temp + output_offset, (*device_output_ptr) + output_offset, output_batch_size * sizeof(float), cudaMemcpyDeviceToHost, A[i]);
    }
    cudaDeviceSynchronize(); // Wait for all streams to complete their operations before proceeding.
    

    for (int x = 0; x < stream_size; x++)
        cudaStreamDestroy(A[x]); // Release allocated resources.

    cudaFree(device_input_ptr);
    cudaFree(device_mask_ptr);
    cudaFree(device_output_ptr);

#undef STREAM_NUM

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }

}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // // Set the kernel dimensions and call the kernel
    // const int Height_out = Height - K + 1;
    // const int Width_out = Width - K + 1;

    // int w_size = ceil((1.0*Width_out)/TILE_WIDTH); 
    // int h_size = ceil((1.0*Height_out)/TILE_WIDTH); 

    // dim3 dimGrid(Map_out, w_size * h_size, ceil((1.0*Batch)/TILE_WIDTH));
    // dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, TILE_WIDTH);
    // conv_forward_kernel<<<dimGrid,dimBlock>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
    
    return;

}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // const int Height_out = Height - K + 1;
    // const int Width_out = Width - K + 1;

    // int out_size = (Height_out*Width_out) * Map_out * Batch * sizeof(float);

    // // Copy the output back to host

    // cudaMemcpy(host_output, device_output, out_size, cudaMemcpyDeviceToHost);

    // // Free device memory

    // cudaFree(device_output);
    // cudaFree(device_input);
    // cudaFree(device_mask);
    return;
    

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
