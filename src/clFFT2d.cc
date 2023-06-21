// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2023 california institute of technology
// all rights reserved

/// File: clFFT2d.cc
/// Desc: openCL FFT2D processor

#include "clFFT2d.h"

#include <cmath>

// constructor, to set all kernels and their args
cl::FFT::FFT2DPlan::FFT2DPlan(clHandle& handle,
    const int width, const int height,
    cl::Buffer& buffer,
    clFFTDirection direction)
{
    cl::Program& program = handle.program;
    CL_CHECK_ERROR(_fft2d_row = cl::Kernel(program, "FFT2D"));
    CL_CHECK_ERROR(_fft2d_col = cl::Kernel(program, "FFT2D"));
    setKernelArgs(handle, width, height, buffer, direction);
}

void cl::FFT::FFT2DPlan::setKernelArgs(clHandle& handle,
    const int width, const int height, cl::Buffer& buffer, clFFTDirection direction)
{
    // check the width and height
    if( !(is_power_of_2(width) && is_power_of_2(height)) ) {
        std::cerr << "width or height needs to be in power of 2, please perform zero-patching \n";
        exit(EXIT_FAILURE);
    }

    // set fft2d_row (along each row) kernel args
    CL_CHECK_ERROR(_fft2d_row.setArg(0, direction));
    CL_CHECK_ERROR(_fft2d_row.setArg(1, width));
    CL_CHECK_ERROR(_fft2d_row.setArg(2, static_cast<cl_int>(std::log2(width))));
    CL_CHECK_ERROR(_fft2d_row.setArg(3, 1)); //stride along row
    CL_CHECK_ERROR(_fft2d_row.setArg(4, buffer));
    CL_CHECK_ERROR(_fft2d_row.setArg(5, cl::Local(width*sizeof(cl_float2))));

    size_type fft2d_maxwg;

    CL_CHECK_ERROR(_fft2d_row.getWorkGroupInfo(handle.device, CL_KERNEL_WORK_GROUP_SIZE, &fft2d_maxwg));
    _fft2d_row_global = cl::NDRange(std::min(static_cast<size_type>(fft2d_maxwg), static_cast<size_type>(width>>1)), static_cast<size_type>(height));
    _fft2d_row_local = cl::NDRange(std::min(static_cast<size_type>(fft2d_maxwg), static_cast<size_type>(width>>1)), 1);

    // set fft2d_col kernel args
        // set fft2d_row (along each row) kernel args
    CL_CHECK_ERROR(_fft2d_col.setArg(0, direction));
    CL_CHECK_ERROR(_fft2d_col.setArg(1, height));
    CL_CHECK_ERROR(_fft2d_col.setArg(2, static_cast<cl_int>(std::log2(height))));
    CL_CHECK_ERROR(_fft2d_col.setArg(3, width)); //stride along column
    CL_CHECK_ERROR(_fft2d_col.setArg(4, buffer));
    CL_CHECK_ERROR(_fft2d_col.setArg(5, cl::Local(height*sizeof(cl_float2))));

    CL_CHECK_ERROR(_fft2d_col.getWorkGroupInfo(handle.device, CL_KERNEL_WORK_GROUP_SIZE, &fft2d_maxwg));
    _fft2d_col_global = cl::NDRange(std::min(static_cast<size_type>(fft2d_maxwg), static_cast<size_type>(height/2)), static_cast<size_type>(width));
    _fft2d_col_local = cl::NDRange(std::min(static_cast<size_type>(fft2d_maxwg), static_cast<size_type>(height/2)), 1);
    // all done
}

/// Execute the FFT
/// @param queue cl Command Queue
/// @param barriers Events need to be finished before executing this
void cl::FFT::FFT2DPlan::execute(cl::CommandQueue& queue,
    const std::vector<cl::Event>* waitlist,
    cl::Event* marker)
{
    // use events to ensure fft2d_col is executed after all fft2d_row processes are done
    // cl::Event event1;

    CL_CHECK_ERROR(queue.enqueueNDRangeKernel(_fft2d_row, cl::NullRange,
        _fft2d_row_global, _fft2d_row_local));
    // std::vector<cl::Event> waitlist1 ={event1};
    CL_CHECK_ERROR(queue.enqueueNDRangeKernel(_fft2d_col, cl::NullRange,
        _fft2d_col_global, _fft2d_col_local));
    // all done
}

// end of file