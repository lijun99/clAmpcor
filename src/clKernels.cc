// collections of ampcor opencl kernels

// enclosed in a string variable

std::string kernelCode = R"(

    #define COS native_cos
    #define SIN native_sin

    //#ifndef FLT_EPSILON
    //    #define FLT_EPSILON 1.19209290E-07F
    //#endif

    // take amplitudes of the complex image
    __kernel void complex_amplitude(
        __global float2* image,
        const int width, const int height // region to take amplitude
        )
    {
        int x = get_global_id(0);
        int y = get_global_id(1);
        int p_width = get_global_size(0);
        int p_height = get_global_size(1);
        int index = y * p_width + x;
        if(x < width && y < height)
        {
            float2 pixel = image[index];
            float amplitude = length(pixel);
            image[index] = (float2)(amplitude, 0.0f);
        }
        else {
            image[index] = (float2)(0.0f, 0.0f);
        }
    }

    // fill the image with 0
    __kernel void complex_fill_zero(
        __global float2* image,
        const int length)
    {
        unsigned int i = get_global_id(0);
        image[i] = (float2)(0.0f, 0.0f);
    }

    __kernel void complex_fill_value(
        __global float2* image,
        const int length,
        const float value)
    {
        unsigned int i = get_global_id(0);
        image[i] = (float2)(value, 0.0f);
    }

    inline float2 complex_mul(const float2 a, const float2 b)
    {
        return (float2)(a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x);
    }

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

    inline float2 complex_mul_conj(const float2 a, const float2 b)
    {
        return (float2)(a.x*b.x+a.y*b.y, a.x*b.y-a.y*b.x);
    }

    __kernel void complex_matrix_element_multiply_conj(
        __global const float2* matrixA,
        __global const float2* matrixB,
        __global float2* result,
        const int width,
        const int height)
    {
        const int x = get_global_id(0);
        const int y = get_global_id(1);

        if (x >= width || y >= height) return;

        const int index = y * width + x;

        result[index] = complex_mul_conj(matrixA[index], matrixB[index]);
    }

    // extract real part from a matrix
    // this kernel is called with globalSize = {out_width, out_height}
     __kernel void matrix_extract_real(
        __global const float2* input,
        __global float2* output,
        const int in_width, const int in_height,
        const int out_width, const int out_height,
        const int origin_x, const int origin_y)
    {
        const int idx = get_global_id(0);
        const int idy = get_global_id(1);

        const int in_idx = idx + origin_x;
        const int in_idy = idy + origin_y;
        const int out_index = idy * out_width + idx;

        if(in_idx>=0 && in_idx<in_width && in_idy>=0 && in_idy<in_height)
        {
            const int in_index = idy* in_width  + idx;
            output[out_index] = (float2)(input[in_index].x, 0.0f);
        }
        else{
            output[out_index] = (float2)(0.0f, 0.0f);
        }
    }

    // fft2d padding zeros in the middle
    // this kernel is called with globalSize = {out_width/2, out_height/2}
    __kernel void fft2d_padding(
        __global const float2* input,
        __global float2* output,
        const int in_width, const int in_height,
        const int out_width, const int out_height)
    {
        const int idx = get_global_id(0);
        const int idy = get_global_id(1);

        // region to copy
        if(idx < in_width/2 && idy < in_height/2)
        {
            // top left quadrature
            output[idy*out_width + idx] = input[idy*in_width + idx];
            // top right quadrature
            output[idy*out_width+(out_width-idx-1)] = input[idy*in_width+(in_width-idx-1)];
            // bottom left quadrature
            output[(out_height-idy-1)*out_width+(idx)] = input[(in_height-idy-1)*in_width+(idx)];
            // bottom right quadrature
            output[(out_height-idy-1)*out_width+(out_width-idx-1)]
                = input[(in_height-idy-1)*in_width+(in_width-idx-1)];
        }
        else { // pad zero
            output[idy*out_width + idx] = (float2)(0.0f, 0.0f);
            output[idy*out_width+(out_width-idx-1)] = (float2)(0.0f, 0.0f);
            output[(out_height-idy-1)*out_width+(idx)] = (float2)(0.0f, 0.0f);
            output[(out_height-idy-1)*out_width+(idx)] = (float2)(0.0f, 0.0f);
        }
    }


    // compute the sum and sum square of a complex image (real part only)
    //  over the region(rx, ry) from the image size (width, height)
    // this kernel is called with workgroupSize = globalSize = {regionx*regiony}
    __kernel void matrix_sum_sum2(
        __global const float2* input, // only sum the real part
        __global float2* sum, // (sum, sum square)
        __local float2* local_sum,
        const int regionx, const int regiony,
        const int width, const int height)
	{
		int globalIndex = get_global_id(0);
		int localIndex = get_local_id(0); // should be the same

		local_sum[localIndex] = (float2)(0.0f, 0.0f); // (x = sum, y=sum square)
        barrier(CLK_LOCAL_MEM_FENCE);

        int n = regionx*regiony;
			// Compute partial sum in each work-item
		while (globalIndex < n)
		{
            int row = globalIndex / regionx;
            int col = globalIndex % regionx;
		    float val = input[row*width+col].x;
			local_sum[localIndex].x += val;
			local_sum[localIndex].y += val*val;
			globalIndex += get_global_size(0);
		}
			// Synchronize all work-items in the work-group
		barrier(CLK_LOCAL_MEM_FENCE);

			// Perform reduction within the work-group
		for (int stride = get_local_size(0) / 2; stride > 0; stride >>= 1)
		{
			if (localIndex < stride)
				local_sum[localIndex] += local_sum[localIndex + stride];

			// Synchronize all work-items in the work-group
			barrier(CLK_LOCAL_MEM_FENCE);
		}

		if (get_local_id(0) == 0)
				sum[0] = local_sum[0];
	}


    // compute the sum area table and sum square of a complex image (real part only)
    __kernel void matrix_sat_sat2(
        __global const float2* input, // only sum the real part
        __global float2* sat2, // (sum, sum square)
        const int width, const int height)
    {

        int globalIndex = get_global_id(0);

        // compute prefix-sum along row at first (each thread for each row)
        // the number of rows may be bigger than the number of threads, iterate
        for (int row = globalIndex; row < height; row += get_global_size(0)) {
            // running sum for value and value^2
            float2 sum2 = (float2)(0.0f, 0.0f);
            int index = row*width;
            // iterative over column
            for (int i=0; i<width; i++) {
                // get the real part
                float val = input[index].x;
                // add it to sum2
                sum2 += (float2)(val, val*val);
                // assign the current accumulated sum to output
                sat2[index] = sum2;
                index++;
            }
        }
        // wait till all sat2
        barrier(CLK_GLOBAL_MEM_FENCE);

        // compute prefix-sum along the row (each thread for each column)
        for (int col = globalIndex; col < width; col += get_global_size(0)) {
            // start position of the current column
            int index = col;
            // assign sum with the first line value
            float2 sum2 = sat2[index];
            // iterative over rest lines
            for (int j=1; j<height; j++) {
                index += width;
                sum2 += sat2[index];
                sat2[index] = sum2;
            }
        }

    } // end of matrix_sat_sat2

    // normalize the correlation surface
    __kernel void correlation_normalize(
        __global float2* surface, // read-write only the real part matters
        __global const float2* referenceSum, // (sum, sum square)
        __global const float2* searchSat, //
        const int regionx, const int regiony, // correlation surface region
        const int window_width, const int window_height,
        const int search_window_width, const int search_window_height // assume the correlation buffer has the same size as searchWindow
        )
    {

        int x = get_global_id(0);
        int y = get_global_id(1);

        // reference
        float2 reference_sum = referenceSum[0];

        // search
        // get four corner at sum area table
        // left top
        float2 lt = (x==0 || y==0) ? (float2)(0.0f, 0.0f) : searchSat[(y-1)*search_window_width+x-1];
        // left bottom
        float2 lb = (x==0) ? (float2)(0.0f, 0.0f) :  searchSat[(y+window_height-1)*search_window_width+x-1];
        // right top
        float2 rt = (y==0) ? (float2)(0.0f, 0.0f) :  searchSat[(y-1)*search_window_width+x+window_width-1];
        // right bottom
        float2 rb = searchSat[(y+window_height-1)*search_window_width+x+window_width-1];
        // get search sum and sum square
        float2 search_sum = rb -lb -rt +lt;
        // normalize cor_norm = (cor_un-norm -<reference><search>)/sqrt( <reference^2> -<reference>^2)(...)
        float size = (float)(window_width*window_height);
        surface[y*search_window_width+x].x -= reference_sum.x * search_sum.x/size;
        surface[y*search_window_width+x].x *= rsqrt(
            (reference_sum.y - reference_sum.x*reference_sum.x/size)
                *(search_sum.y-search_sum.x*search_sum.x/size)+FLT_EPSILON);
    } // end of correlation_normalize

    // find the max (real part) location on an image
    //  over the region(rx, ry) from the image size (width, height)
    // this kernel is called with workgroupSize = globalSize = {regionx*regiony}
    __kernel void matrix_max_location(
        __global const float2* input, // only sum the real part
        __global int2* maxloc, // along (width, height)
        __local float* local_max, // local memory to save max value and location
        __local int* local_maxloc,
        const int regionx, const int regiony,
        const int width, const int height)
	{
		int globalIndex = get_global_id(0);
		int localIndex = get_local_id(0); // should be the same
        int groupSize = get_local_size(0);

		local_max[localIndex] = 0.0f; // (assume amplitudes are positive)
        local_maxloc[localIndex] = 0;

	    // Compute partial sum in each work-item
		for (int id = globalIndex; id < regionx*regiony; id += groupSize)
		{
            int row = id / regionx;
            int col = id % regionx;
		    float val = input[row*width+col].x;
			if(local_max[localIndex] < val) {
			    local_max[localIndex] = val;
			    local_maxloc[localIndex] = id;
			}
		}
		// Synchronize all work-items in the work-group
		barrier(CLK_LOCAL_MEM_FENCE);

			// Perform reduction within the work-group
		for (int stride = groupSize / 2; stride > 0; stride >>= 1)
		{
			if (localIndex < stride)
			{
			    if(local_max[localIndex + stride] > local_max[localIndex])
			    {
			        local_max[localIndex] = local_max[localIndex + stride];
			        local_maxloc[localIndex] = local_maxloc[localIndex + stride];
			    }
			}
			// Synchronize all work-items in the work-group
			barrier(CLK_LOCAL_MEM_FENCE);
		}
        // use the thread 0 to return the result
		if (localIndex == 0) {
		    maxloc[0].x = local_maxloc[0]%regionx;
		    maxloc[0].y = local_maxloc[0]/regionx; // (col, row)
		}
	}


)";
