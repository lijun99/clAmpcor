
#include "clHelper.h"

std::ostream& operator<<(std::ostream& os, const cl_int2& vec) {
    os << "(" << vec.x << ", " << vec.y << ")";
    return os;
}

std::ostream& operator<<(std::ostream& os, const cl_float2& vec) {
    os << "(" << vec.x << ", " << vec.y << ")";
    return os;
}

cl_int2 make_int2(const int&a, const int& b)
{
    cl_int2 c;
    c.x = a;
    c.y = b;
    return c;
}

template <typename T>
void buffer_debug(cl::CommandQueue& queue, cl::Buffer& buffer,
    const int width, const int height, std::string str)
{
    T*  mappedPtr = new T[width*height];
    CL_CHECK_ERROR(queue.enqueueReadBuffer(
        buffer, // buffer (device)
        CL_TRUE, // blocking
        0, // offset
        width*height*sizeof(T),
        mappedPtr));

        std::cout << str << "\n";
        for(int j=0; j<height; j++) {
            if (j<ELEMENTS_TO_SHOW || height-j < ELEMENTS_TO_SHOW) {
                for(int i=0; i<width; i++)
                {
                    if(i<ELEMENTS_TO_SHOW || width-i < ELEMENTS_TO_SHOW)
                        std::cout  << mappedPtr[j*width+i] << " ";
                }
                std::cout << "\n";
            }
        }
}

// explicit instantiations
template void buffer_debug<cl_float2>(cl::CommandQueue& queue, cl::Buffer& buffer,
    const int width, const int height, std::string str);
template void buffer_debug<cl_int2>(cl::CommandQueue& queue, cl::Buffer& buffer,
    const int width, const int height, std::string str);
template void buffer_debug<cl_float>(cl::CommandQueue& queue, cl::Buffer& buffer,
    const int width, const int height, std::string str);
template void buffer_debug<cl_int>(cl::CommandQueue& queue, cl::Buffer& buffer,
    const int width, const int height, std::string str);

template <typename T>
void buffer_print(cl::CommandQueue& queue, cl::Buffer& buffer,
    const int width, const int height, std::string str)
{
    std::cout << str << "\n";
    std::vector<T> mappedPtr(width*height);
    CL_CHECK_ERROR(queue.enqueueReadBuffer(
        buffer, // buffer (device)
        CL_TRUE, // blocking
        0, // offset
        width*height*sizeof(T),
        mappedPtr.data()));

    for(int j=0; j<height; j++) {
        for(int i=0; i< width; i++) {
            std::cout  << mappedPtr[j*width+i] << " ";
        }
        std::cout << "\n";
    }
}

// explicit instantiations
template void buffer_print<cl_float2>(cl::CommandQueue& queue, cl::Buffer& buffer,
    const int width, const int height, std::string str);
template void buffer_print<cl_int2>(cl::CommandQueue& queue, cl::Buffer& buffer,
    const int width, const int height, std::string str);
template void buffer_print<cl_float>(cl::CommandQueue& queue, cl::Buffer& buffer,
    const int width, const int height, std::string str);
template void buffer_print<cl_int>(cl::CommandQueue& queue, cl::Buffer& buffer,
    const int width, const int height, std::string str);


bool is_power_of_2(const ::size_t n)
{
    return n && !(n & (n - 1));
}

cl::size_type next_power_of_2(const int n)
{
    cl::size_type r = 1;
    while (r<n)
        r<<=1;
    return r;
}

cl::Program buildCLProgramFromString(cl::Context& context, std::string& source)
{
    // initiate the program
    cl::Program program(context, source);
    // build cl kernels and check errors (at runtime)
    try {
        program.build(CL_AMPCOR_BUILD_OPTIONS);
    } catch (const cl::Error& e) {
        if (e.err() == CL_BUILD_PROGRAM_FAILURE) {
            // Print the build log if there was an error
            cl::string buildLog;
            program.getBuildInfo(context.getInfo<CL_CONTEXT_DEVICES>()[0], CL_PROGRAM_BUILD_LOG, &buildLog);
            std::cerr << "Build log:\n" << buildLog << std::endl;
        }
        throw e;
    }
    // upon success, return
    return program;
}

clHandle::clHandle() {
    initialize();
}

clHandle::clHandle(cl_device_type devType) : deviceType(devType){
    initialize();
}

void clHandle::initialize()
{
    // get platforms
    CL_CHECK_ERROR(cl::Platform::get(&platforms));
    if (platforms.empty()) {
        std::cerr << "No OpenCL platforms found!" << std::endl;
        exit(EXIT_FAILURE);
    }
    // get gpu devices
    CL_CHECK_ERROR(platforms[0].getDevices(deviceType, &devices));
    if (devices.empty()) {
        std::cerr << "No OpenCL Devices found!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // set up context
    CL_CHECK_ERROR(context = cl::Context(devices));
    // use device 0 as default
    setDevice(0);
}

void clHandle::setDevice(const int devID)
{
    device = devices[devID];
}


char *getCLErrorString(cl_int err){

    switch (err) {
        case CL_SUCCESS:                          return (char *) "Success!";
        case CL_DEVICE_NOT_FOUND:                 return (char *) "No OpenCL devices that matched the specified criteria were found.";
        case CL_DEVICE_NOT_AVAILABLE:             return (char *) "The device is currently unavailable.";
        case CL_COMPILER_NOT_AVAILABLE:           return (char *) "Compiler not available";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:    return (char *) "Memory object allocation failure";
        case CL_OUT_OF_RESOURCES:                 return (char *) "Out of resources";
        case CL_OUT_OF_HOST_MEMORY:               return (char *) "Out of host memory";
        case CL_PROFILING_INFO_NOT_AVAILABLE:     return (char *) "Profiling information not available";
        case CL_MEM_COPY_OVERLAP:                 return (char *) "Memory copy overlap";
        case CL_IMAGE_FORMAT_MISMATCH:            return (char *) "Image format mismatch";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:       return (char *) "Image format not supported";
        case CL_BUILD_PROGRAM_FAILURE:            return (char *) "Program build failure";
        case CL_MAP_FAILURE:                      return (char *) "Map failure";
        case CL_INVALID_VALUE:                    return (char *) "One or more argument values are invalid";
        case CL_INVALID_DEVICE_TYPE:              return (char *) "Invalid device type";
        case CL_INVALID_PLATFORM:                 return (char *) "Invalid platform";
        case CL_INVALID_DEVICE:                   return (char *) "Invalid device";
        case CL_INVALID_CONTEXT:                  return (char *) "Invalid context";
        case CL_INVALID_QUEUE_PROPERTIES:         return (char *) "Invalid queue properties";
        case CL_INVALID_COMMAND_QUEUE:            return (char *) "Invalid command queue";
        case CL_INVALID_HOST_PTR:                 return (char *) "Invalid host pointer";
        case CL_INVALID_MEM_OBJECT:               return (char *) "Invalid memory object";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:  return (char *) "Invalid image format descriptor";
        case CL_INVALID_IMAGE_SIZE:               return (char *) "Invalid image size";
        case CL_INVALID_SAMPLER:                  return (char *) "Invalid sampler";
        case CL_INVALID_BINARY:                   return (char *) "Invalid binary";
        case CL_INVALID_BUILD_OPTIONS:            return (char *) "Invalid build options";
        case CL_INVALID_PROGRAM:                  return (char *) "Invalid program";
        case CL_INVALID_PROGRAM_EXECUTABLE:       return (char *) "Invalid program executable";
        case CL_INVALID_KERNEL_NAME:              return (char *) "Invalid kernel name";
        case CL_INVALID_KERNEL_DEFINITION:        return (char *) "Invalid kernel definition";
        case CL_INVALID_KERNEL:                   return (char *) "Invalid kernel";
        case CL_INVALID_ARG_INDEX:                return (char *) "Invalid argument index";
        case CL_INVALID_ARG_VALUE:                return (char *) "Invalid argument value";
        case CL_INVALID_ARG_SIZE:                 return (char *) "Invalid argument size";
        case CL_INVALID_KERNEL_ARGS:              return (char *) "Invalid kernel arguments";
        case CL_INVALID_WORK_DIMENSION:           return (char *) "Invalid work dimension";
        case CL_INVALID_WORK_GROUP_SIZE:          return (char *) "Invalid work group size";
        case CL_INVALID_WORK_ITEM_SIZE:           return (char *) "Invalid work item size";
        case CL_INVALID_GLOBAL_OFFSET:            return (char *) "Invalid global offset";
        case CL_INVALID_EVENT_WAIT_LIST:          return (char *) "Invalid event wait list";
        case CL_INVALID_EVENT:                    return (char *) "Invalid event";
        case CL_INVALID_OPERATION:                 return (char *) "Invalid operation";
        case CL_INVALID_GL_OBJECT:                return (char *) "Invalid OpenGL object";
        case CL_INVALID_BUFFER_SIZE:              return (char *) "Invalid buffer size";
        case CL_INVALID_MIP_LEVEL:                return (char *) "Invalid mip-map level";
        default:                                  return (char *) "Unknown";
    }
}
//end of file
