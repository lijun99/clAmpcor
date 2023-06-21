// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2023 california institute of technology
// all rights reserved

/// @file clOversampler.h
/// @breif openCL oversampling a complex/real image, using FFT method
///
/// The steps are 1) FFT the image to frequency space;
/// 2) enlarge the frequency image to a larger size,
///    moving originals to four corners (freq shift), and padding 0
/// 3) FFT the enlarged image back to real space

// guard
#pragma once
// dependencies
#include "clHelper.h"
#include "clFFT2d.h"

namespace cl { namespace Ampcor {

class Oversampler {

public:
    using size_type = cl::size_type;
    using complex_type = cl_float2;
    using float_type = cl_float;
    using int_type = cl_int;
    using fft_plan_type = cl::FFT::FFT2DPlan;
    using kernel_type = cl::Kernel;

    // methods
    // Oversampler () = default;
    Oversampler(clHandle& handle,
        const int in_width, const int in_height,
        const int out_width, const int out_height,
        cl::Buffer& input, cl::Buffer& output);
    ~Oversampler() = default;
    void setKernelArgs(clHandle& handle,
        const int in_width, const int in_height,
        const int out_width, const int out_height,
        cl::Buffer& input, cl::Buffer& output);
    void execute(cl::CommandQueue& queue,
        const std::vector<cl::Event>* waitlist = nullptr,
        cl::Event* marker=nullptr);

private:
    fft_plan_type _forward_fft;
    fft_plan_type _inverse_fft;
    cl::Buffer& _input;
    cl::Buffer& _output;
    int _in_width;
    int _in_height;
    int _out_width;
    int _out_height;
    kernel_type _matrix_fft_padding;
    cl::NDRange _matrix_fft_padding_global;

};

}} // end of namespace

