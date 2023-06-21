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

// my definition
#include "clOversampler.h"

// constructor, to set all kernels and their args
cl::Ampcor::Oversampler::Oversampler(clHandle& handle,
    const int in_width, const int in_height, const int out_width, const int out_height,
    cl::Buffer& input, cl::Buffer& output)
    : _input(input), _output(output), _in_width(in_width), _in_height(in_height),
    _out_width(out_width), _out_height(out_height)
{
    setKernelArgs(handle, in_width, in_height, out_width, out_height, input, output);
}

void cl::Ampcor::Oversampler::setKernelArgs(clHandle& handle,
    const int in_width, const int in_height, const int out_width, const int out_height,
    cl::Buffer& input, cl::Buffer& output)

{
    _forward_fft = fft_plan_type(handle, in_width, in_height, input, CL_FFT_FORWARD);
    _inverse_fft = fft_plan_type(handle, out_width, out_height, output, CL_FFT_INVERSE);

    // grab the padding kernel
    CL_CHECK_ERROR(_matrix_fft_padding = cl::Kernel(handle.program, "matrix_fft_padding"));
    // Set kernel arguments
    int argIndex = 0;
    CL_CHECK_ERROR(_matrix_fft_padding.setArg(argIndex++, input));
    CL_CHECK_ERROR(_matrix_fft_padding.setArg(argIndex++, output));
    CL_CHECK_ERROR(_matrix_fft_padding.setArg(argIndex++, in_width));
    CL_CHECK_ERROR(_matrix_fft_padding.setArg(argIndex++, in_height));
    CL_CHECK_ERROR(_matrix_fft_padding.setArg(argIndex++, out_width));
    CL_CHECK_ERROR(_matrix_fft_padding.setArg(argIndex++, out_height));
    // set global size
    _matrix_fft_padding_global = cl::NDRange(out_width >> 1, out_height >> 1);
    // all done
}

void cl::Ampcor::Oversampler::execute(cl::CommandQueue& queue,
    const std::vector<cl::Event>* waitlist,
    cl::Event* marker)
{

#ifdef CL_AMPCOR_STEP_DEBUG
    std::cout << "debug oversampler " << _in_width << _in_height << "\n";
            buffer_debug<cl_float2>(queue, _input,
                _in_width, _in_height, "oversampler input before fft");
#endif
    // fft input to freq space
    _forward_fft.execute(queue);

#ifdef CL_AMPCOR_STEP_DEBUG
            buffer_debug<cl_float2>(queue, _input,
                _in_width, _in_height, "oversampler input after fft");
#endif

    // padding
    CL_CHECK_ERROR(queue.enqueueNDRangeKernel(
        _matrix_fft_padding,
        cl::NullRange,
        _matrix_fft_padding_global
        ));

#ifdef CL_AMPCOR_STEP_DEBUG
            buffer_debug<cl_float2>(queue, _output,
                _out_width, _out_height, "oversampler padded");
#endif

    // fft correlation surface back to real space
    _inverse_fft.execute(queue);
    // all done
}

