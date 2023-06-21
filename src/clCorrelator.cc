// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2023 california institute of technology
// all rights reserved

/// File: clCorrelator.h
/// Desc: openCL Cross-Correlation processor, using FFT method
///  C(x, y) = \sum_{X,Y} R(X, Y) S(X+x, Y+y) = IFFT[(FFT[R])^* dotprod FFT[S]]

// my definition
#include "clCorrelator.h"

// constructor, to set all kernels and their args
cl::Ampcor::Correlator::Correlator(clHandle& handle,
    const int width, const int height,
    cl::Buffer& reference, cl::Buffer& secondary, cl::Buffer& correlation)
{
    setKernelArgs(handle, width, height, reference, secondary, correlation);
}

void cl::Ampcor::Correlator::setKernelArgs(clHandle& handle,
    const int width, const int height,
    cl::Buffer& reference, cl::Buffer& secondary, cl::Buffer& correlation)

{
   _reference_fft = fft_plan_type(handle, width, height, reference, CL_FFT_FORWARD);
   _secondary_fft = fft_plan_type(handle, width, height, secondary, CL_FFT_FORWARD);
   _correlation_fft = fft_plan_type(handle, width, height, correlation, CL_FFT_INVERSE);

   CL_CHECK_ERROR(_matrix_mul_conj = cl::Kernel(handle.program, "matrix_element_multiply_conj"));
    // Set kernel arguments
    int argIndex = 0;
    CL_CHECK_ERROR(_matrix_mul_conj.setArg(argIndex++, reference));
    CL_CHECK_ERROR(_matrix_mul_conj.setArg(argIndex++, secondary));
    CL_CHECK_ERROR(_matrix_mul_conj.setArg(argIndex++, correlation));
    CL_CHECK_ERROR(_matrix_mul_conj.setArg(argIndex++, width));
    CL_CHECK_ERROR(_matrix_mul_conj.setArg(argIndex++, height));

    _matrix_mul_conj_global = cl::NDRange(width, height);
    // all done
}

void cl::Ampcor::Correlator::execute(cl::CommandQueue& queue,
    const std::vector<cl::Event>* waitlist,
    cl::Event* marker)
{
    // fft reference to freq space
    _reference_fft.execute(queue);
    // fft secondary to freq space
    _secondary_fft.execute(queue);
    // conjugate multiply to get correlation
    CL_CHECK_ERROR(queue.enqueueNDRangeKernel(
        _matrix_mul_conj,
        cl::NullRange,
        _matrix_mul_conj_global
        ));
    // fft correlation surface back to real space
    _correlation_fft.execute(queue);
    // all done
}

