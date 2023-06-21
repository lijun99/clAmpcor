// shell command wrapper

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include "clHelper.h"
#include "clFFT2d.h"
#include "clProgram.h"

void deviceQuery(clHandle& handle);
void fft2dTest(clHandle& handle);


int main() {

    // ******* OpenCL initialization *********
    // initialize the opencl handles
    clHandle handle(CL_DEVICE_TYPE_GPU);
    // build the kernel program
    handle.program = cl::Ampcor::Program(handle.context);

    // run tests
    deviceQuery(handle);
    fft2dTest(handle);
    // all done
    return 0;
}

void deviceQuery(clHandle& handle)
{
    cl::Device& device = handle.device;
    std::cout << "Testing Device Information ...... \n";
    std::cout << "Number of Devices: " << handle.devices.size() << std::endl;
    std::cout << "Device Type: " << device.getInfo<CL_DEVICE_TYPE>() << std::endl;
    std::cout << "Device 0: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    std::cout << "Vendor: " << device.getInfo<CL_DEVICE_VENDOR>() << std::endl;
    std::cout << "OpenCL Version: " << device.getInfo<CL_DEVICE_VERSION>() << std::endl;
    std::cout << "Max Compute Units: " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
    std::cout << "Max Clock Frequency(MHz): " << device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << std::endl;
    std::cout << "Max Work Group Size: " << device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;
    std::vector<::size_t> maxWorkItemSizes;
    device.getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &maxWorkItemSizes);
    std::cout << "Max Work Item Sizes: (" << maxWorkItemSizes[0] << ", "
        << maxWorkItemSizes[1] << ", "
        << maxWorkItemSizes[2] << ")"
        << std::endl;
    std::cout << "Preferred Vector Width (float): " << device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT>()  << std::endl;
    std::cout << "Local Memory Size: " << device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() / 1024 << " KB" << std::endl;
    std::cout << "Global Memory Size: " << device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() / (1024 * 1024) << " MB" << std::endl;
    std::cout << "OpenCL Extensions: " << device.getInfo<CL_DEVICE_EXTENSIONS>() << std::endl;
    // all done
    std::cout << std::endl;
}

void fft2dTest(clHandle& handle)
{
    std::cout << "Testing FFT2D ......\n";

    // get references for cl handles
    cl::Context& context = handle.context;
    cl::Device& device = handle.device;
    cl::Program& program = handle.program;

    // create a command queue
    cl::CommandQueue queue(context, device);

    // Create input and output vectors
    const int width = 8;
    const int height = 4;
    const size_t nsize = sizeof(cl_float2)*width*height;

    // Create OpenCL device buffers for the input and output vectors
    cl::Buffer bufferA(context, CL_MEM_READ_WRITE, nsize);

    // initialize and set values on host
    std::vector<cl_float2> A(width*height);
    for (int i = 0; i < height; i++) {
        for(int j=0; j < width; j++)
        {
            int id = i*width +j;
            A[id].x =   (float)j/width;
            A[id].y = 0.0f;
        }
    }

    // create the FFT2d plan
    cl::FFT::FFT2DPlan fft2d(handle, width, height, bufferA,
        CL_FFT_FORWARD);
    cl::FFT::FFT2DPlan ifft2d(handle, width, height, bufferA,
        CL_FFT_INVERSE);

    // copy data to device
    CL_CHECK_ERROR(queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, nsize, A.data()));
    // fft
    fft2d.execute(queue);
    // check results
    buffer_print<cl_float2>(queue, bufferA, width, height, "after fft" );

    // inverse fft
    ifft2d.execute(queue);
    // check results
    buffer_print<cl_float2>(queue, bufferA, width, height, "after ifft" );

    std::random_device rd;
    std::mt19937 engine(rd());
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    // change the values to matrix
    for (auto& ai : A) {
        ai.x = distribution(engine);
        ai.y = 0.0f;
    }
    // repeat fft
    CL_CHECK_ERROR(queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, nsize, A.data()));
    buffer_print<cl_float2>(queue, bufferA, width, height, "a new random matrix" );
    fft2d.execute(queue);
    buffer_print<cl_float2>(queue, bufferA, width, height, "after fft w/ new matrix" );
    // inverse fft
    ifft2d.execute(queue);
    // check results
    buffer_print<cl_float2>(queue, bufferA, width, height, "after ifft (not normalized)" );
    // all done

    queue.finish();
}
