// OpenCL FFT2D Kernel code
// Cooley-Tukey Radix-2 algorithm

std::string FFT2d_CL_code = R"(

    __attribute__((always_inline))
    unsigned int bit_reverse(unsigned int num, int bits)
    {
        unsigned int reversed_num = 0;
        int i;

        for (i = 0; i < bits; i++)
        {
            reversed_num <<= 1;
            reversed_num |= (num & 1);
            num >>= 1;
        }
        return reversed_num;
    }

    // Perform in-place FFT for a 2D complex matrix
    // this kernel needs to called twice, one along row and one along column
    // length(width or height) needs to be in power of 2
    __kernel void FFT2D(
        int direction, // 1 = forward, -1 = inverse
        int length, // width or height
        int log2_length, // log2(width) or log2(height)
        int stride, // 1 along row, width along column
        __global float2* matrix,
        __local float4* smem)
    {
        int local_id = get_local_id(0);
        int local_size = get_local_size(0);
        float fdirection = (float)direction;

        int offset = (stride==1) ? get_group_id(1) << log2_length : get_group_id(1);
        int half_size = 1;
        int log2_half_size = 0;
        int half_length = (length >> 1);

        // first radix-2
        for (int i=local_id; i < half_length; i+=local_size)
        {
            // get even and odd indices
            int k00 = i << 1;
            int k01 = k00 + 1;

            // use bit reverse to arrange the order
            int index0 = bit_reverse(k00, log2_length);
            int index1 = bit_reverse(k01, log2_length);

            // get input, stride=1 for along row, stride=width for along column
            float2 in_data_0 = matrix[mad24(index0, stride, offset)];
            float2 in_data_1 = matrix[mad24(index1, stride, offset)];

            // set the output, use float4 to vectorize the operation
            float4 out_data;
            out_data.x = in_data_0.x + in_data_1.x;
            out_data.y = in_data_0.y + in_data_1.y;
            out_data.z = in_data_0.x - in_data_1.x;
            out_data.w = in_data_0.y - in_data_1.y;

            smem[i] = out_data;
        }
        // set a barrier to wait all elements are processed
        barrier(CLK_LOCAL_MEM_FENCE);

        // iterate radix-2, with data saved in local memory
        half_size = half_size << 1;
        log2_half_size++;
        for (int iteration = 1; iteration < log2_length - 1; iteration++)
        {

            for (int i = local_id; i < (half_length >> 1); i += local_size)
            {
                int bufferfly_size = 2 * half_size;
                int k = (2 * i) & (half_size - 1);
                int bfly_offset = ((2 * i) >> log2_half_size)*bufferfly_size;

                // Compute the butterflies
                int k00 = bfly_offset + k;
                int k01 = k00 + half_size;

                float recip_bufferfly_size = native_recip((float)bufferfly_size);
                float twiddle_x = native_cos(2.0f * M_PI_F * k * recip_bufferfly_size);
                float twiddle_y = fdirection * native_sin(2.0f * M_PI_F * k * recip_bufferfly_size);

                float4 in_data = smem[k01 >> 1];

                float tmp0 = twiddle_x * in_data.x + twiddle_y * in_data.y;
                float tmp1 = twiddle_x * in_data.y - twiddle_y * in_data.x;

                twiddle_x = native_cos(2.0f * M_PI_F * (k + 1) * recip_bufferfly_size);
                twiddle_y = fdirection * native_sin(2.0f * M_PI_F * (k + 1) * recip_bufferfly_size);

                float tmp2 = twiddle_x * in_data.z + twiddle_y * in_data.w;
                float tmp3 = twiddle_x * in_data.w - twiddle_y * in_data.z;

                in_data = smem[k00 >> 1];
                float4 out_data;

                out_data.x = in_data.x - tmp0;
                out_data.y = in_data.y - tmp1;
                out_data.z = in_data.z - tmp2;
                out_data.w = in_data.w - tmp3;

                smem[k01 >> 1] = out_data;

                out_data.x = in_data.x + tmp0;
                out_data.y = in_data.y + tmp1;
                out_data.z = in_data.z + tmp2;
                out_data.w = in_data.w + tmp3;

                smem[k00 >> 1] = out_data;
            }
            // set a barrier to synchronize each iteration
            barrier(CLK_LOCAL_MEM_FENCE);

            half_size = half_size << 1;
            log2_half_size++;
        }

        // last radix-2 iteration, writing data back to matrix
        for (int i = local_id; i < (half_length)>>1; i+=local_size)
        {
            int bufferfly_size = 2 * half_size;
            int k = (2 * i) & (half_size - 1);
            int bfly_offset = ((2 * i) >> log2_half_size)*bufferfly_size;

            // Compute the butterflies
            int k00 = bfly_offset + k;
            int k01 = k00 + half_size;

            float recip_bufferfly_size = native_recip((float)bufferfly_size);
            float twiddle_x = native_cos(2.0f * M_PI_F * k * recip_bufferfly_size);
            float twiddle_y = fdirection * native_sin(2.0f * M_PI_F * k * recip_bufferfly_size);

            float4 in_data = smem[k01 >> 1];

            float tmp0 = twiddle_x * in_data.x + twiddle_y * in_data.y;
            float tmp1 = twiddle_x * in_data.y - twiddle_y * in_data.x;

            twiddle_x = native_cos(2.0f * M_PI_F * (k + 1) * recip_bufferfly_size);
            twiddle_y = fdirection * native_sin(2.0f * M_PI_F * (k + 1) * recip_bufferfly_size);

            float tmp2 = twiddle_x * in_data.z + twiddle_y * in_data.w;
            float tmp3 = twiddle_x * in_data.w - twiddle_y * in_data.z;

            float2 out_data;

            in_data = smem[k00 >> 1];
            out_data.x = in_data.x - tmp0;
            out_data.y = in_data.y - tmp1;
            matrix[mad24(k01, stride, offset)] = out_data;

            out_data.x = in_data.z - tmp2;
            out_data.y = in_data.w - tmp3;
            matrix[mad24(k01+1, stride, offset)] = out_data;

            out_data.x = in_data.x + tmp0;
            out_data.y = in_data.y + tmp1;
            matrix[mad24(k00, stride, offset)] = out_data;

            out_data.x = in_data.z + tmp2;
            out_data.y = in_data.w + tmp3;
            matrix[mad24(k00+1, stride, offset)] = out_data;

        }
        // all done
    }

)";
// end of file
