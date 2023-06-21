///
/// @file Kernels.cc
/// @brief A collection of ampcor opencl kernels
///

// CL Kernels enclosed in a string variable
std::string Matrix_CL_code = R"(

    // take amplitudes in a rect region (width, length) of the complex image
    //   with size (p_width, p_height);
    // set zeros to the rest
    __kernel void matrix_complex_amplitude(
        __global float2* image,
        const int width, const int height, // work region
        const int p_width, const int p_height // whole image
        )
    {
        int col = get_global_id(0);
        int row = get_global_id(1);

        int index =  mad24(row, p_width, col);
        if(row < height && col < width)
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
    __kernel void matrix_fill_zero(
        __global float2* image)
    {
        const int i = get_global_id(0);
        image[i] = (float2)(0.0f, 0.0f);
    }

    __kernel void complex_fill_value(
        __global float2* image,
        const int length,
        const float value)
    {
        const int i = get_global_id(0);
        image[i] = (float2)(value, 0.0f);
    }

    // width / height is the actual buffer size
    // actual rectangle area is controlled by global_size(0) (1)
    __kernel void matrix_element_multiply_conj(
        __global const float2* matrixA,
        __global const float2* matrixB,
        __global float2* result,
        const int width,
        const int height)
    {
        const int col = get_global_id(0);
        const int row = get_global_id(1);
        const int index = mad24(row, width, col);

        result[index] = complex_mul_conj(matrixA[index], matrixB[index]);
    }

    // extract real part from a matrix
    // this kernel is called with globalSize = {out_width, out_height}
     __kernel void matrix_extract_real(
        __global const float2* input,
        __global float2* output,
        const int in_width, const int in_height,
        const int in_stride,
        __global const int2* max_loc,
        const int offsetx, const int offsety)
    {
        const int idx = get_global_id(0);
        const int idy = get_global_id(1);
        const int out_width = get_global_size(0);

        const int in_idx = idx + max_loc[0].x + offsetx;
        const int in_idy = idy + max_loc[0].y + offsety;

        if(in_idx>=0 && in_idx<in_width && in_idy>=0 && in_idy<in_height)
        {
            output[mad24(idy, out_width, idx)] = (float2)(input[mad24(in_idy, in_stride, in_idx)].x, 0.0f);
        }
        else{
            output[mad24(idy, out_width, idx)] = (float2)(0.0f, 0.0f);
        }
    }

    // fft2d padding zeros in the middle
    // this kernel is called with globalSize = {out_width/2, out_height/2}
    __kernel void matrix_fft_padding(
        __global const float2* input,
        __global float2* output,
        const int in_width, const int in_height,
        const int out_width, const int out_height)
    {
        const int idx = get_global_id(0);
        const int idy = get_global_id(1);

        const int half_in_width = in_width >> 1;
        const int half_in_height = in_height >> 1;

        // region to copy
        if(idx < half_in_width && idy < half_in_height)
        {
            // top left quadrature
            output[mad24(idy, out_width, idx)] = input[mad24(idy, in_width, idx)];
            // top right quadrature
            output[mad24(idy, out_width, out_width-idx-1)]
                = input[mad24(idy, in_width, in_width-idx-1)];
            // bottom left quadrature
            output[mad24(out_height-idy-1, out_width, idx)] = input[mad24(in_height-idy-1, in_width, idx)];
            // bottom right quadrature
            output[mad24(out_height-idy-1, out_width, out_width-idx-1)]
                = input[mad24(in_height-idy-1, in_width, in_width-idx-1)];
        }
        else { // pad zero
            output[mad24(idy, out_width,  idx)] = (float2)(0.0f, 0.0f);
            output[mad24(idy, out_width, out_width-idx-1)] = (float2)(0.0f, 0.0f);
            output[mad24(out_height-idy-1, out_width, idx)] = (float2)(0.0f, 0.0f);
            output[mad24(out_height-idy-1, out_width, out_width-idx-1)] = (float2)(0.0f, 0.0f);
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
		for (int i = globalIndex; i < n; i+=get_global_size(0))
		{
            int row = i / regionx;
            int col = i % regionx;
		    float val = input[row*width+col].x;
			local_sum[localIndex].x += val;
			local_sum[localIndex].y += val*val;
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
        const int width, const int height, // region and output
        const int p_width, const int p_height) // storage dimension of input
    {

        int globalIndex = get_global_id(0);

        // compute prefix-sum along row at first (each thread for each row)
        // the number of rows may be bigger than the number of threads, iterate
        for (int row = globalIndex; row < height; row += get_global_size(0)) {
            // running sum for value and value^2
            float2 sum2 = (float2)(0.0f, 0.0f);
            int index = row * p_width;
            int out_index = row * width;
            // iterative over column
            for (int i=0; i<width; i++) {
                // get the real part
                float val = input[index].x;
                // add it to sum2
                sum2 += (float2)(val, val*val);
                // assign the current accumulated sum to output
                sat2[out_index] = sum2;
                index++;
                out_index++;
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
        //barrier(CLK_GLOBAL_MEM_FENCE);

    } // end of matrix_sat_sat2

    // normalize the correlation surface
    __kernel void correlation_normalize(
        __global float2* surface, // read-write only the real part matters
        __global const float2* referenceSum, // (sum, sum square)
        __global const float2* searchSat, //
        const int regionx, const int regiony, // correlation surface region
        const int storage_width, const int storage_height, // matrix size for storing the correlation surface
        const int window_width, const int window_height, // reference window size
        const int search_window_width, const int search_window_height // search window size
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
        float size_recip = native_recip((float)(window_width*window_height));
        float size_recip2 = native_recip((float)(storage_width*storage_height)); //fft norm
        // only normalize real part
        const int index = mad24(y, storage_width, x);
        float temp = surface[index].x*size_recip2 - reference_sum.x * search_sum.x*size_recip;
        //printf("debug norm %d %g %g %g %g %g %g\n", index, search_sum.x, reference_sum.x, surface[index].x, temp,
        //    reference_sum.y, search_sum.y);
        temp *= native_rsqrt(
            (reference_sum.y - reference_sum.x*reference_sum.x*size_recip)
                *(search_sum.y-search_sum.x*search_sum.x*size_recip)+FLT_EPSILON);
        surface[index].x = temp;
    } // end of correlation_normalize

    // find the max (real part) location on an image
    //  over the region(rx, ry) from the image size (width, height)
    // this kernel is called with workgroupSize = globalSize = {regionx*regiony}
    __kernel void matrix_max_location(
        __global const float2* input, // only sum the real part
        __global int2* maxloc, // along (width, height)
        __local float* local_max, // local memory to save max value and location
        __local int* local_maxloc,
        const int width, const int height,
        const int stride)
	{
		int globalIndex = get_global_id(0);
		int localIndex = get_local_id(0); // should be the same as globalIndex
        int groupSize = get_local_size(0);

		local_max[localIndex] = 0.0f; // (assume amplitudes are positive)
        local_maxloc[localIndex] = 0;

	    // Compute partial sum in each work-item
		for (int id = globalIndex; id < width*height; id += groupSize)
		{
		    int col = id % width;
            int row = id / width;
		    float val = input[mad24(row, stride, col)].x;
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
		    maxloc[0].x = local_maxloc[0] % width;
		    maxloc[0].y = local_maxloc[0] / width; // (col, row)
		}
	}

    __kernel void matrix_transpose(
        const uint rows,
        const uint cols,
        __global float2* matrix,
        __global float2* matrixTranspose)
    {
        const uint i = get_global_id(0) << 1;
        const uint j = get_global_id(1) << 1;

        float4 temp = *(__global float4*)&matrix[mad24(j, cols, i)];
        float4 temp1 = *(__global float4*)&matrix[mad24(j, cols, i) + cols];

        *(__global float4*)&matrixTranspose[mad24(i, rows, j)] = (float4)(temp.s01, temp1.s01);
        *(__global float4*)&matrixTranspose[mad24(i, rows, j) + rows] = (float4)(temp.s23, temp1.s23);
    }

)";

// end of file