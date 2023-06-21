// OpenCL Common Code for other Kernels

std::string Common_CL_code = R"(

    #define COS native_cos
    #define SIN native_sin

    //#ifndef FLT_EPSILON
    //    #define FLT_EPSILON 1.19209290E-07F
    //#endif
)";
// end of file