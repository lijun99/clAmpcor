// OpenCL FFT2D Kernel code - Simple Formula

std::string FFT2d_simple_CL_code = R"(

    // OpenCL kernel for 2D Fast Fourier Transform (FFT) along columns
    __kernel void fft2d(__global const float2* input,
        __global float2* output,
        const int width, const int height)
    {
        int x = get_global_id(0);  // Column index
        int y = get_global_id(1);  // Row index

        float2 sum = {0.0f, 0.0f};
        for (int i = 0; i< width ; i++)
        {
            float var = 2.0f * M_PI_F * x * i / (float)width;
            float2 twiddlex = (float2)(COS(var), -SIN(var));

            for (int j = 0; j < height; j++)
            {
                var = 2.0f * M_PI_F * y * j / (float)height;
                float2 twiddley = (float2)(COS(var), -SIN(var));
                float2 twiddle = complex_mul(twiddlex, twiddley);
                sum +=  complex_mul(input[j * width + i], twiddle);
            }
        }
        output[y * width + x] = sum;
    }

        // OpenCL kernel for 2D Fast Fourier Transform (FFT) along columns
    __kernel void ifft2d(__global const float2* input,
        __global float2* output,
        const int width, const int height)
    {
        int x = get_global_id(0);  // Column index
        int y = get_global_id(1);  // Row index

        float2 sum = {0.0f, 0.0f};
        for (int i = 0; i< width ; i++)
        {
            float var = 2.0f * M_PI_F * x * i / (float)width;
            float2 twiddlex = (float2)(COS(var), SIN(var));

            for (int j = 0; j < height; j++)
            {
                var = 2.0f * M_PI_F * y * j / (float)height;
                float2 twiddley = (float2)(COS(var), SIN(var));
                float2 twiddle = complex_mul(twiddlex, twiddley);
                sum +=  complex_mul(input[j * width + i], twiddle);
            }
        }
        output[y * width + x] = sum/(float)(height*width);
    }

)";
