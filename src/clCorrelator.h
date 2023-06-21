// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2023 california institute of technology
// all rights reserved

/// File: clCorrelator.h
/// Desc: openCL Cross-Correlation processor, using FFT method
///  C(x, y) = \sum_{X,Y} R(X, Y) S(X+x, Y+y) = IFFT[(FFT[R])^* dotprod FFT[S]]

// guard
#pragma once

// dependencies
#include "clHelper.h"
#include "clFFT2d.h"

namespace cl { namespace Ampcor {

class Correlator {

public:
    using size_type = cl::size_type;
    using complex_type = cl_float2;
    using float_type = cl_float;
    using int_type = cl_int;
    using fft_plan_type = cl::FFT::FFT2DPlan;
    using kernel_type = cl::Kernel;

    // methods
    Correlator () = default;
    Correlator(clHandle& handle,
        const int width, const int height,
        cl::Buffer& reference,
        cl::Buffer& secondary,
        cl::Buffer& correlation);
    ~Correlator() = default;
    void setKernelArgs(clHandle& handle,
        const int width, const int height,
        cl::Buffer& reference,
        cl::Buffer& secondary,
        cl::Buffer& correlation);
    void execute(cl::CommandQueue& queue,
        const std::vector<cl::Event>* waitlist = nullptr,
        cl::Event* marker=nullptr);

private:
    fft_plan_type _reference_fft;
    fft_plan_type _secondary_fft;
    fft_plan_type _correlation_fft;
    kernel_type _matrix_mul_conj;

    cl::NDRange _matrix_mul_conj_global;

};

}} // end of namespace

