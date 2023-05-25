
#include "clHelper.h"

char *getCLErrorString(cl_int err){

    switch (err) {
        case CL_SUCCESS:                          return (char *) "Success!";
        case CL_DEVICE_NOT_FOUND:                 return (char *) "Device not found.";
        case CL_DEVICE_NOT_AVAILABLE:             return (char *) "Device not available";
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

 /**************************
        CL_SUCCESS (0): Success
        CL_DEVICE_NOT_FOUND (-1): No OpenCL devices that matched the specified criteria were found
        CL_DEVICE_NOT_AVAILABLE (-2): The device is currently unavailable
        CL_COMPILER_NOT_AVAILABLE (-3): The compiler is not available
        CL_MEM_OBJECT_ALLOCATION_FAILURE (-4): Failed to allocate memory on the device
        CL_OUT_OF_RESOURCES (-5): Failed to allocate resources on the device
        CL_OUT_OF_HOST_MEMORY (-6): Failed to allocate resources on the host
        CL_PROFILING_INFO_NOT_AVAILABLE (-7): Profiling information is not available
        CL_MEM_COPY_OVERLAP (-8): The source and destination memory regions overlap
        CL_IMAGE_FORMAT_MISMATCH (-9): The image format is not compatible with the device
        CL_IMAGE_FORMAT_NOT_SUPPORTED (-10): The image format is not supported by the device
        CL_BUILD_PROGRAM_FAILURE (-11): Failed to build the program executable
        CL_MAP_FAILURE (-12): Failed to map the requested region into the host address space
        CL_MISALIGNED_SUB_BUFFER_OFFSET (-13): The sub-buffer offset is not aligned to CL_DEVICE_MEM_BASE_ADDR_ALIGN
        CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST (-14): An event in the wait list returned an error status
        CL_COMPILE_PROGRAM_FAILURE (-15): Failed to compile the program source code
        CL_LINKER_NOT_AVAILABLE (-16): The linker is not available
        CL_LINK_PROGRAM_FAILURE (-17): Failed to link the compiled program object(s)
        CL_DEVICE_PARTITION_FAILED (-18): Failed to partition the device
        CL_KERNEL_ARG_INFO_NOT_AVAILABLE (-19): Kernel argument information is not available
        CL_INVALID_VALUE (-30): One or more argument values are invalid
        CL_INVALID_DEVICE_TYPE (-31): The device type specified is not valid
        CL_INVALID_PLATFORM (-32): The platform specified is not valid
        CL_INVALID_DEVICE (-33): The device specified is not valid
        CL_INVALID_CONTEXT (-34): The context specified is not valid
        CL_INVALID_QUEUE_PROPERTIES (-35): The specified queue properties are not supported by the device
        CL_INVALID_COMMAND_QUEUE (-36): The command queue specified is not valid
        CL_INVALID_HOST_PTR (-37): The host pointer specified is not valid
        CL_INVALID_MEM_OBJECT (-38): The memory object specified is not valid
        CL_INVALID_IMAGE_FORMAT_DESCRIPTOR (-39): The image format specified is not valid
        CL_INVALID_IMAGE_SIZE (-40): The specified image size is not valid
        CL_INVALID_SAMPLER (-41): The specified sampler is not valid
        CL_INVALID_BINARY (-42): The binary program is not valid
        CL_INVALID_BUILD_OPTIONS (-43): The specified build options are not valid
        CL_INVALID_PROGRAM (-44): The specified program is not valid
        CL_INVALID_PROGRAM_EXECUTABLE (-45): The program executable specified is not valid
        CL_INVALID_KERNEL_NAME (-46): The kernel name specified is not valid
        CL_INVALID_KERNEL_DEFINITION (-47): The kernel definition specified is not valid
        CL_INVALID_KERNEL (-48): The kernel specified is not valid
        CL_INVALID_ARG_INDEX (-49): The argument index specified is not valid
        CL_INVALID_ARG_VALUE (-50): The argument
    ***********************/
