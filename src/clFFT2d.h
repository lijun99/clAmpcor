// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2023 california institute of technology
// all rights reserved

/// @file clFFT2d.h
/// @brief openCL FFT2D processor

// guard
#pragma once

#include "clHelper.h"

enum clFFTDirection {
    CL_FFT_FORWARD = 1,
    CL_FFT_INVERSE = -1
};

namespace cl { namespace FFT {

class FFT2DPlan {
public:
    using size_type = cl::size_type;
    using complex_type = cl_float2;
    using float_type = cl_float;
    using int_type = cl_int;

    // methods
    FFT2DPlan() = default;
    FFT2DPlan(clHandle& handle,
        const int width, const int height,
        cl::Buffer& buffer,
        clFFTDirection direction=CL_FFT_FORWARD);
    ~FFT2DPlan() = default;
    void setKernelArgs(clHandle& handle,
        const int width, const int height,
        cl::Buffer& buffer,
        clFFTDirection direction);
    void execute(cl::CommandQueue& queue,
        const std::vector<cl::Event>* waitlist = nullptr,
        cl::Event* marker=nullptr);

private:
    // variables
    clFFTDirection _direction;
    cl::Kernel _fft2d_row;
    cl::Kernel _fft2d_col;
    cl::NDRange _fft2d_row_global;
    cl::NDRange _fft2d_row_local;
    cl::NDRange _fft2d_col_global;
    cl::NDRange _fft2d_col_local;

};

} } // end of namespace cl
// end of file