// OpenCL for Complex/float2 operations

std::string Complex_CL_code = R"(

    __attribute__((always_inline))
    float2 complex_mul(const float2 a, const float2 b)
    {
        float2 result;
        result.x = fma(a.x, b.x, -a.y * b.y);
        result.y = fma(a.x, b.y, a.y * b.x);
        return result;
    }

    __attribute__((always_inline))
    float2 complex_mul_conj(const float2 a, const float2 b)
    {
        float2 result;
        result.x = fma(a.x, b.x, a.y * b.y);
        result.y = fma(a.x, b.y, -a.y * b.x);
        return result;
    }



)";
// end of file
