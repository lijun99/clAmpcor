/// @file ReduceCL2.cc
/// @brief Group Reduction for devices with OpenCL 2.0 support

std::string Reduce_CL2_code = R"(

    //
    void atomic_add_f(volatile global float* addr, const float val) {
        private float old, sum;
        do {
            old = *addr;
            sum = old+val;
        } while(atomic_cmpxchg((volatile global int*)addr, as_int(old), as_int(sum))!=as_int(old));
    }


    __kernel void sum_kernel(__global const float* a, __global float* sum)
    {
        // sum within a work group
        float group_sum = work_group_reduce_add(a[get_global_id(0)]);
        // use atomic to sum over different groups
        if (get_local_id(0) == 0)
            atomic_add_f(b, group_sum);
    }


)";

// end of file
