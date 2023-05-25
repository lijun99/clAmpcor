#include "clAmpcor.h"

#include <iostream>
#include <fstream>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

// the cl Kernel source codes
#include "clKernels.cc"


// for debugging

std::ostream& operator<<(std::ostream& os, const cl_float2& vec) {
    os << "(" << vec.x << ", " << vec.y << ")";
    return os;
}

cl_int2 make_int2(const int&a, const int& b)
{
    cl_int2 c;
    c.x = a;
    c.y = b;
    return c;
}

std::ostream& operator<<(std::ostream& os, const cl_int2& vec) {
    os << "(" << vec.x << ", " << vec.y << ")";
    return os;
}

template <typename T>
void buffer_debug(cl::CommandQueue& queue, cl::Buffer& buffer,
    const int width, const int height, std::string str)
{
    T*  mappedPtr = new T[width*height];
    CL_CHECK_ERROR(queue.enqueueReadBuffer(
        buffer, // buffer (device)
        CL_TRUE, // blocking
        0, // offset
        width*height*sizeof(T),
        mappedPtr));

        std::cout << str << "\n";
        for(int i=0; i< std::min(10, width); i++)
            std::cout  << mappedPtr[i] << " ";
        std::cout << "\n";
        for(int i=std::max(width-10, 0); i< width; i++)
            std::cout  << mappedPtr[(height-1)*width+i] << " ";
        std::cout << "\n";
}

template <typename T>
void buffer_print(cl::CommandQueue& queue, cl::Buffer& buffer,
    const int width, const int height, std::string str)
{
    std::cout << str << "\n";
    std::vector<T> mappedPtr(width*height);
    CL_CHECK_ERROR(queue.enqueueReadBuffer(
        buffer, // buffer (device)
        CL_TRUE, // blocking
        0, // offset
        width*height*sizeof(T),
        mappedPtr.data()));

    for(int j=0; j<height; j++) {
        for(int i=0; i< width; i++) {
            std::cout  << mappedPtr[j*width+i] << " ";
        }
        std::cout << "\n";
    }

}

bool is_power_of_2(const ::size_t n)
{
    return n && !(n & (n - 1));
}

cl::size_type next_power_of_2(const int n)
{
    cl::size_type r = 1;
    while (r<n)
        r<<=1;
    return r;
}


void cl::Ampcor::read_parameters_from_json(const std::string& filename)
{
    std::ifstream settings_file (filename);
    std::string line;

    if (settings_file.fail()){
        std::cerr << "Config file does not exist.";
        exit(EXIT_FAILURE);
    }

    json settings;

    try {
        settings = json::parse(settings_file);
    }
    catch (const json::parse_error& e) {
        std::cerr << "JSON parse error: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    try {
        // Access the values existing in JSON settings
        referenceImageName = settings.at("reference").value("slc", "reference.slc");
        referenceImageWidth = settings.at("reference").value("width", 0);
        referenceImageHeight = settings.at("reference").value("height", 0);
        secondaryImageName = settings.at("secondary").value("slc", "secondary.slc");
        secondaryImageWidth = settings.at("secondary").value("width", 0);
        secondaryImageHeight = settings.at("secondary").value("height", 0);

        offsetImageName = settings.at("offset").value("slc", "offset.slc");

        windowWidthRaw = settings.at("window").value("width", 64);
        windowHeightRaw = settings.at("window").value("height", 64);

        halfSearchRangeAcrossRaw = settings.at("half_search_range").value("across", 20);
        halfSearchRangeDownRaw = settings.at("half_search_range").value("down", 20);

        // set the search window size accordingly
        searchWindowWidthRaw = windowWidthRaw + halfSearchRangeAcrossRaw*2;
        searchWindowHeightRaw = windowHeightRaw + halfSearchRangeDownRaw*2;

        skipSampleAcross = settings.at("skip_between_windows").value("across", 128);
        skipSampleDown = settings.at("skip_between_windows").value("down", 128);

        secondaryStartPixelAcross = settings.at("start_pixel_secondary").value("across", searchWindowWidthRaw/2);
        secondaryStartPixelDown = settings.at("start_pixel_secondary").value("down", searchWindowHeightRaw/2);

        secondaryEndPixelAcross = settings.at("end_pixel_secondary").value("across",
            secondaryImageWidth-searchWindowWidthRaw/2);
        secondaryEndPixelDown = settings.at("end_pixel_secondary").value("down",
            secondaryImageHeight-searchWindowHeightRaw/2);

        numberWindowAcross = settings.at("offset").value("width", 0);
        numberWindowDown = settings.at("offset").value("height", 0);

        // adjust the number of windows if necessary
        int_type nWin = (secondaryEndPixelAcross-secondaryStartPixelAcross)/skipSampleAcross;
        if (numberWindowAcross <= 0)
            numberWindowAcross = nWin;
        else
            numberWindowAcross = std::min(numberWindowAcross, nWin);

        nWin = (secondaryEndPixelDown-secondaryStartPixelDown)/skipSampleDown;
        if (numberWindowDown <= 0)
            numberWindowDown = nWin;
        else
            numberWindowDown = std::min(numberWindowDown, nWin);

        windowWidth = windowWidthRaw*rawDataOversamplingFactor;
        windowHeight = windowHeightRaw*rawDataOversamplingFactor;
        searchWindowWidth = searchWindowWidthRaw*rawDataOversamplingFactor;
        searchWindowHeight = searchWindowHeightRaw*rawDataOversamplingFactor;

        correlationSurfaceWidth = searchWindowWidth - windowWidth + 1;
        correlationSurfaceHeight = searchWindowHeight - windowHeight + 1;

        halfZoomWindowSizeRaw = settings.at("correlation_surface_zoom_in").value(
                "half_range", std::min(halfSearchRangeAcrossRaw, 4));

        halfZoomWindowSizeRaw = std::min(halfZoomWindowSizeRaw,
            std::min(correlationSurfaceWidth, correlationSurfaceHeight)/2);

        oversamplingFactor = settings.at("correlation_surface_zoom_in").value(
                "oversampling_factor", 32);

        zoomWindowSize = 2*halfZoomWindowSizeRaw;
        correlationSurfaceSizeOversampled = zoomWindowSize*oversamplingFactor;

        std::cout << "Processing Ampcor between " << referenceImageName
            << " and " << secondaryImageName << "\n"
            << "Template window size "
                << make_int2(windowWidth, windowHeight) << "\n"
            << "Search window size "
                << make_int2(searchWindowWidth, searchWindowHeight) << "\n"
            << "starting pixel (center of the first search window)"
                << make_int2(secondaryStartPixelAcross, secondaryStartPixelDown) << "\n"
            << "number of windows "
                << make_int2(numberWindowAcross, numberWindowDown) << "\n" << "\n";

    } catch (const json::type_error& e) {
        std::cerr << "JSON type error: " << e.what() << std::endl;
        return;
    } catch (const json::out_of_range& e) {
        std::cerr << "JSON out of range error: " << e.what() << std::endl;
        return;
    }


}

void cl::Ampcor::run()
{
    // read settings
    read_parameters_from_json("ampcor.json");

    // open reference and secondary image files
    std::ifstream referenceFile(referenceImageName, std::ios::binary);
    if (referenceFile.fail()){
        std::cerr << "The reference image file does not exist. \n";
        exit(EXIT_FAILURE);
    }
    std::ifstream secondaryFile(secondaryImageName, std::ios::binary);
    if (secondaryFile.fail()){
        std::cerr << "The secondary image file does not exist. \n";
        exit(EXIT_FAILURE);
    }

    // ******* OpenCL initialization *********
    // Get available OpenCL platforms
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    // Get the first available OpenCL device on the first platform
    std::vector<cl::Device> devices;
    platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

    // Create an OpenCL context and command queue
    cl::Context context(devices);
    cl::Device device = devices[0];
    cl::CommandQueue queue(context, device);

    // get the max work group size - to be used with local memory reduce
    size_type maxWorkGroupSize =device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();

    // print out GPU information
    std::cout << "Device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    std::cout << "Vendor: " << device.getInfo<CL_DEVICE_VENDOR>() << std::endl;
    std::cout << "OpenCL Version: " << device.getInfo<CL_DEVICE_VERSION>() << std::endl;
    std::cout << "Max Compute Units (not ALUs): " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
    std::cout << "Max Work Group Size: " << device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;
    std::cout << "Local Memory Size: " << device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() / 1024 << " KB" << std::endl;
    std::cout << "Global Memory Size: " << device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() / (1024 * 1024) << " MB" << std::endl;
    std::cout << std::endl;


    // ******* CPU/host Buffers *************
    // Create read buffers for reference images
    size_type referenceBufferSize = referenceImageWidth*windowHeightRaw*cfloatBytes;
    char * referenceBufferHost = new char[referenceBufferSize];

    // Create read buffers for secondary images
    size_type secondaryBufferSize = secondaryImageWidth*searchWindowHeightRaw*cfloatBytes;
    char * secondaryBufferHost = new char[secondaryBufferSize];

    // offset image
    cl_float2* offset_image = new cl_float2[numberWindowAcross*numberWindowDown];
    cl_int2 offsetRaw, offsetFrac;

    // ******** GPU/device Buffers ***************

    // reference image (windowWidth, windowHeight), but enlarged to the search window size
    cl::Buffer referenceWindow(context, CL_MEM_READ_WRITE,
        searchWindowWidth*searchWindowHeight*cfloatBytes);
    // search image, window + search range
    cl::Buffer searchWindow(context, CL_MEM_READ_WRITE,
        searchWindowWidth*searchWindowHeight*cfloatBytes);

    // temp buffer for fft cross-correlation
    cl::Buffer fftCorrWork(context, CL_MEM_READ_WRITE,
        searchWindowWidth*searchWindowHeight*cfloatBytes);

    // reference image sum and sum square
    cl::Buffer referenceWindowSum2(context, CL_MEM_READ_WRITE,
        cfloatBytes);
    // secondary image sum area table
    cl::Buffer searchWindowSAT2(context, CL_MEM_READ_WRITE,
        searchWindowWidth*searchWindowHeight*cfloatBytes);


    // correlation surfaces
    cl::Buffer correlationSurfaceRaw(context, CL_MEM_READ_WRITE,
        searchWindowWidth*searchWindowHeight*cfloatBytes);
    cl::Buffer correlationSurfaceZoom(context, CL_MEM_READ_WRITE,
        zoomWindowSize*zoomWindowSize*cfloatBytes);
    cl::Buffer correlationSurfaceZoomF(context, CL_MEM_READ_WRITE,
        zoomWindowSize*zoomWindowSize*cfloatBytes);
    cl::Buffer correlationSurfaceOSF(context, CL_MEM_READ_WRITE,
        correlationSurfaceSizeOversampled*correlationSurfaceSizeOversampled*cfloatBytes);
    cl::Buffer correlationSurfaceOS(context, CL_MEM_READ_WRITE,
        correlationSurfaceSizeOversampled*correlationSurfaceSizeOversampled*cfloatBytes);

    // correlation surface max location/offset
    cl::Buffer corrSurfaceMaxLoc(context, CL_MEM_READ_WRITE,
        sizeof(cl_int2));

    // ************ cl kernels **********************
    //if build kernels from an external file, this file needs to be explicitly provided with shell command
    //cl::Program program = buildProgramFromSource(context, "ampcor.cl");

    // get the kernel code from included clKernels.h
    cl::Program program(context, kernelCode);
    // build cl kernels and check errors (at runtime)
    try {
        program.build("-cl-std=CL1.2");
    } catch (const cl::Error& e) {
        if (e.err() == CL_BUILD_PROGRAM_FAILURE) {
            // Print the build log if there was an error
            cl::string buildLog;
            program.getBuildInfo(context.getInfo<CL_CONTEXT_DEVICES>()[0], CL_PROGRAM_BUILD_LOG, &buildLog);
            std::cerr << "Build log:\n" << buildLog << std::endl;
        }
        throw e;
    }
    // get kernels from the program
    cl::Kernel amplitudeKernel(program, "complex_amplitude");
    cl::Kernel referenceSumKernel(program, "matrix_sum_sum2");
    cl::Kernel searchSatKernel(program, "matrix_sat_sat2");
    cl::Kernel fft2dKernel(program, "fft2d");
    cl::Kernel ifft2dKernel(program, "ifft2d");
    cl::Kernel elementMultiplyKernel(program, "complex_matrix_element_multiply_conj");
    cl::Kernel corrNormalizeKernel(program, "correlation_normalize");
    cl::Kernel extractRealKernel(program, "matrix_extract_real");
    cl::Kernel cmatrixSetValue(program, "complex_fill_value");
    cl::Kernel findMaxLocationKernel(program, "matrix_max_location");
    cl::Kernel fft2dPaddingKernel(program, "fft2d_padding");

    cl::NDRange globalSize;
    size_type groupSize;

    // ************* Processing ************
    // message interval
    int_type message_interval = std::max(numberWindowDown/10, 1);
    // iterative over windows along height
    for(int_type iWindowDown=0; iWindowDown<numberWindowDown; iWindowDown++)
    {
        // **** read image to buffers
        // determine the starting line(s)
        size_type secondaryLineStart = secondaryStartPixelDown - searchWindowHeightRaw/2
            + iWindowDown*skipSampleDown;
        size_type referenceLineStart = secondaryLineStart + halfSearchRangeDownRaw;
        // load the reference buffer
        std::streampos offset;
        offset = referenceLineStart*referenceImageWidth*cfloatBytes ;
        referenceFile.seekg(offset);
        referenceFile.read(referenceBufferHost, referenceBufferSize);
        // load the secondary buffer
        offset = secondaryLineStart*secondaryImageWidth*cfloatBytes;
        secondaryFile.seekg(offset);
        secondaryFile.read(secondaryBufferHost, secondaryBufferSize);

        if(iWindowDown%message_interval == 0)
            std::cout << "Processing windows (" << iWindowDown <<", x) - ("
                << std::min(numberWindowDown, iWindowDown+message_interval)
                << ", x) out of " << numberWindowDown << std::endl;

        // iterate over windows along width
        for(int_type iWindowAcross = 0; iWindowAcross<numberWindowAcross; iWindowAcross++)
        {

            // determine the starting column (along width)
            size_type secondaryColStart = secondaryStartPixelAcross - searchWindowWidthRaw/2
                + iWindowAcross*skipSampleAcross;
            size_type referenceColStart = secondaryColStart + halfSearchRangeAcrossRaw;

            // std::cout << "referenceStart " << referenceColStart << " " << referenceLineStart << "\n";
            //  std::cout << "secondaryStart " << secondaryColStart << " " << secondaryLineStart << "\n";

            cl::size_t<3> s_origin;
            s_origin[0] = referenceColStart*cfloatBytes;
            s_origin[1] = 0;
            s_origin[2] = 0;

            cl::size_t<3> d_origin;
            d_origin[0] = 0;
            d_origin[1] = 0;
            d_origin[2] = 0;

            cl::size_t<3> region;
            region[0] = windowWidth*cfloatBytes;
            region[1] = windowHeight;
            region[2] = 1;


            // copy a window from reference host to device buffer
            queue.enqueueWriteBufferRect(
                referenceWindow, // buffer
                CL_TRUE, // blocking
                d_origin, // buffer origin
                s_origin, // host origin
                region,   // rect region
                searchWindowWidth*cfloatBytes,       // dst buffer_row_pitch
                0,    // buffer_slice_pitch, n/a for 1d/2d
                referenceImageWidth*cfloatBytes,       // host_row_pitch
                0,    // host_slice_pitch
                referenceBufferHost // host posize_typeer
                );

            // take amplitude
            CL_CHECK_ERROR(amplitudeKernel.setArg(0, referenceWindow));
            CL_CHECK_ERROR(amplitudeKernel.setArg(1, windowWidth));
            CL_CHECK_ERROR(amplitudeKernel.setArg(2, windowHeight));
            cl::NDRange globalSize(searchWindowWidth, searchWindowHeight);
            queue.enqueueNDRangeKernel(amplitudeKernel, cl::NullRange, globalSize);

            //buffer_print<cl_float2>(queue, referenceWindow,
            //    searchWindowWidth, searchWindowHeight, "reference amplitude");

            // compute the sum and sum square of reference window - for normalization
            groupSize = std::min(next_power_of_2(windowHeight*windowWidth), maxWorkGroupSize);
            CL_CHECK_ERROR(referenceSumKernel.setArg(0, referenceWindow));
            CL_CHECK_ERROR(referenceSumKernel.setArg(1, referenceWindowSum2));
            CL_CHECK_ERROR(referenceSumKernel.setArg(2, cl::Local(cfloatBytes*groupSize)));
            CL_CHECK_ERROR(referenceSumKernel.setArg(3, windowWidth));
            CL_CHECK_ERROR(referenceSumKernel.setArg(4, windowHeight));
            CL_CHECK_ERROR(referenceSumKernel.setArg(5, searchWindowWidth));
            CL_CHECK_ERROR(referenceSumKernel.setArg(6, searchWindowHeight));
            queue.enqueueNDRangeKernel(referenceSumKernel, cl::NullRange,
                cl::NDRange(groupSize), // globalSize
                cl::NDRange(groupSize));  // local/Workgroup Size

            //buffer_print<cl_float2>(queue, referenceWindowSum2,
            //    1, 1, "reference sum");


            // copy a window from secondary buffer
            s_origin[0] = secondaryColStart*cfloatBytes;
            region[0] = searchWindowWidthRaw*cfloatBytes;
            region[1] = searchWindowHeightRaw;
            queue.enqueueWriteBufferRect(
                searchWindow, // buffer
                CL_TRUE, // blocking
                d_origin, // buffer origin
                s_origin, // host origin
                region,   // rect region
                searchWindowWidth*cfloatBytes,       // dst buffer_row_pitch
                0,    // buffer_slice_pitch, n/a for 1d/2d
                secondaryImageWidth*cfloatBytes,       // host_row_pitch
                0,    // host_slice_pitch
                secondaryBufferHost // host posize_typeer
                );


            // take the amplitude
            CL_CHECK_ERROR(amplitudeKernel.setArg(0, searchWindow));
            CL_CHECK_ERROR(amplitudeKernel.setArg(1, searchWindowWidth));
            CL_CHECK_ERROR(amplitudeKernel.setArg(2, searchWindowHeight));
            globalSize=cl::NDRange(searchWindowWidth, searchWindowHeight);
            queue.enqueueNDRangeKernel(amplitudeKernel,
                cl::NullRange,  // work_group/localSize
                cl::NDRange(searchWindowWidth, searchWindowHeight) //globalSize
                );

            //buffer_print<cl_float2>(queue, searchWindow,
            //    searchWindowWidth, searchWindowHeight, "searchWindow amplitude");

            // compute the sum area table
            groupSize = std::min(static_cast<size_type>(std::max(searchWindowWidth, searchWindowHeight)),
                maxWorkGroupSize);

            CL_CHECK_ERROR(searchSatKernel.setArg(0, searchWindow));
            CL_CHECK_ERROR(searchSatKernel.setArg(1, searchWindowSAT2));
            CL_CHECK_ERROR(searchSatKernel.setArg(2, searchWindowWidth));
            CL_CHECK_ERROR(searchSatKernel.setArg(3, searchWindowHeight));
            CL_CHECK_ERROR(queue.enqueueNDRangeKernel(searchSatKernel, cl::NullRange,
                cl::NDRange(groupSize), // globalSize
                cl::NDRange(groupSize)));  // local/Workgroup Size

            //buffer_print<cl_float2>(queue, searchWindowSAT2,
            //    searchWindowWidth, searchWindowHeight, "search sat");

            // fft to frequency space for reference window
            CL_CHECK_ERROR(fft2dKernel.setArg(0, referenceWindow));
            CL_CHECK_ERROR(fft2dKernel.setArg(1, fftCorrWork));
            CL_CHECK_ERROR(fft2dKernel.setArg(2, searchWindowWidth));
            CL_CHECK_ERROR(fft2dKernel.setArg(3, searchWindowHeight));
            queue.enqueueNDRangeKernel(fft2dKernel, cl::NullRange, globalSize);
            // copy back to referenceWindow
            queue.enqueueCopyBuffer(fftCorrWork, referenceWindow, // src, dst
                0, 0, // src_offset, dst_offset in bytes
                searchWindowWidth*searchWindowHeight*cfloatBytes // size in bytes
                );

            //buffer_print<cl_float2>(queue, referenceWindow,
            //    searchWindowWidth, searchWindowHeight, "reference fft");

            // fft to frequency space for reference window
            CL_CHECK_ERROR(fft2dKernel.setArg(0, searchWindow));
            CL_CHECK_ERROR(fft2dKernel.setArg(1, fftCorrWork));
            CL_CHECK_ERROR(fft2dKernel.setArg(2, searchWindowWidth));
            CL_CHECK_ERROR(fft2dKernel.setArg(3, searchWindowHeight));
            queue.enqueueNDRangeKernel(fft2dKernel, cl::NullRange, globalSize);
            // copy back to referenceWindow
            queue.enqueueCopyBuffer(fftCorrWork, searchWindow, // src, dst
                0, 0, // src_offset, dst_offset in bytes
                searchWindowWidth*searchWindowHeight*cfloatBytes // size in bytes
                );

            //buffer_print<cl_float2>(queue, searchWindow,
            //    searchWindowWidth, searchWindowHeight, "searchWindow fft");

            // multiply elements of (enlarged) window and searchWindow in frequency space
            CL_CHECK_ERROR(elementMultiplyKernel.setArg(0, referenceWindow));
            CL_CHECK_ERROR(elementMultiplyKernel.setArg(1, searchWindow));
            CL_CHECK_ERROR(elementMultiplyKernel.setArg(2, fftCorrWork));
            CL_CHECK_ERROR(elementMultiplyKernel.setArg(3, searchWindowWidth));
            CL_CHECK_ERROR(elementMultiplyKernel.setArg(4, searchWindowHeight));
            queue.enqueueNDRangeKernel(elementMultiplyKernel, cl::NullRange, globalSize);

            //buffer_print<cl_float2>(queue, fftCorrWork,
            //    searchWindowWidth, searchWindowHeight, "correlation fft");

            // inverse fft correlationSurface
            CL_CHECK_ERROR(ifft2dKernel.setArg(0, fftCorrWork));
            CL_CHECK_ERROR(ifft2dKernel.setArg(1, correlationSurfaceRaw));
            CL_CHECK_ERROR(ifft2dKernel.setArg(2, searchWindowWidth));
            CL_CHECK_ERROR(ifft2dKernel.setArg(3, searchWindowHeight));
            queue.enqueueNDRangeKernel(ifft2dKernel, cl::NullRange, globalSize);

            //buffer_print<cl_float2>(queue, correlationSurfaceRaw,
            //    searchWindowWidth, searchWindowHeight, "correlation large");


            // normalize the correlation surface
            CL_CHECK_ERROR(corrNormalizeKernel.setArg(0, correlationSurfaceRaw));
            CL_CHECK_ERROR(corrNormalizeKernel.setArg(1, referenceWindowSum2));
            CL_CHECK_ERROR(corrNormalizeKernel.setArg(2, searchWindowSAT2));
            CL_CHECK_ERROR(corrNormalizeKernel.setArg(3, correlationSurfaceWidth));
            CL_CHECK_ERROR(corrNormalizeKernel.setArg(4, correlationSurfaceHeight));
            CL_CHECK_ERROR(corrNormalizeKernel.setArg(5, windowWidth));
            CL_CHECK_ERROR(corrNormalizeKernel.setArg(6, windowHeight));
            CL_CHECK_ERROR(corrNormalizeKernel.setArg(7, searchWindowWidth));
            CL_CHECK_ERROR(corrNormalizeKernel.setArg(8, searchWindowHeight));
            globalSize = cl::NDRange(correlationSurfaceWidth, correlationSurfaceHeight);
            queue.enqueueNDRangeKernel(corrNormalizeKernel, cl::NullRange, globalSize);

            // buffer_print<cl_float2>(queue, correlationSurfaceRaw,
            //    searchWindowWidth, searchWindowHeight, "correlation normalized");

            // find the max location in correlation surface
            groupSize = std::min(next_power_of_2(correlationSurfaceWidth*correlationSurfaceHeight),
                maxWorkGroupSize);
            CL_CHECK_ERROR(findMaxLocationKernel.setArg(0, correlationSurfaceRaw));
            CL_CHECK_ERROR(findMaxLocationKernel.setArg(1, corrSurfaceMaxLoc));
            CL_CHECK_ERROR(findMaxLocationKernel.setArg(2, cl::Local(groupSize*sizeof(float))));
            CL_CHECK_ERROR(findMaxLocationKernel.setArg(3, cl::Local(groupSize*sizeof(int))));
            CL_CHECK_ERROR(findMaxLocationKernel.setArg(4, correlationSurfaceWidth));
            CL_CHECK_ERROR(findMaxLocationKernel.setArg(5, correlationSurfaceHeight));
            CL_CHECK_ERROR(findMaxLocationKernel.setArg(6, searchWindowWidth));
            CL_CHECK_ERROR(findMaxLocationKernel.setArg(7, searchWindowHeight));
            CL_CHECK_ERROR(queue.enqueueNDRangeKernel(findMaxLocationKernel, cl::NullRange,
                cl::NDRange(groupSize), // globalSize
                cl::NDRange(groupSize)));  // local/Workgroup Size

            // copy max location
            CL_CHECK_ERROR(queue.enqueueReadBuffer(
                corrSurfaceMaxLoc,
                CL_TRUE, // blocking
                0, // offset
                sizeof(cl_int2),
                &offsetRaw));

            // std::cout << "max location first pass " << offsetRaw << "\n";
            offsetRaw.x -= halfZoomWindowSizeRaw;
            offsetRaw.y -= halfZoomWindowSizeRaw;
            // std::cout << "max location first pass " << offsetRaw << "\n";

            // extract the real part and the top corners to get the correlation surface
            CL_CHECK_ERROR(extractRealKernel.setArg(0, correlationSurfaceRaw));
            CL_CHECK_ERROR(extractRealKernel.setArg(1, correlationSurfaceZoom));
            CL_CHECK_ERROR(extractRealKernel.setArg(2, searchWindowWidth));
            CL_CHECK_ERROR(extractRealKernel.setArg(3, searchWindowHeight));
            CL_CHECK_ERROR(extractRealKernel.setArg(4, zoomWindowSize));
            CL_CHECK_ERROR(extractRealKernel.setArg(5, zoomWindowSize));
            CL_CHECK_ERROR(extractRealKernel.setArg(6, offsetRaw.x));
            CL_CHECK_ERROR(extractRealKernel.setArg(7, offsetRaw.y));
            globalSize = cl::NDRange(zoomWindowSize, zoomWindowSize);
            queue.enqueueNDRangeKernel(extractRealKernel, cl::NullRange, globalSize);

            //buffer_print<cl_float2>(queue, correlationSurfaceZoom,
            //    zoomWindowSize, zoomWindowSize, "correlationSurfaceZoom");

            /// use fft to oversample the correlation surface
            // fft the correlation surface to frequency space
            CL_CHECK_ERROR(fft2dKernel.setArg(0, correlationSurfaceZoom));
            CL_CHECK_ERROR(fft2dKernel.setArg(1, correlationSurfaceZoomF));
            CL_CHECK_ERROR(fft2dKernel.setArg(2, zoomWindowSize));
            CL_CHECK_ERROR(fft2dKernel.setArg(3, zoomWindowSize));
            queue.enqueueNDRangeKernel(fft2dKernel, cl::NullRange,
                cl::NDRange(zoomWindowSize, zoomWindowSize));

            // padding zeros in frequency space
            CL_CHECK_ERROR(fft2dPaddingKernel.setArg(0, correlationSurfaceZoomF));
            CL_CHECK_ERROR(fft2dPaddingKernel.setArg(1, correlationSurfaceOSF));
            CL_CHECK_ERROR(fft2dPaddingKernel.setArg(2, zoomWindowSize));
            CL_CHECK_ERROR(fft2dPaddingKernel.setArg(3, zoomWindowSize));
            CL_CHECK_ERROR(fft2dPaddingKernel.setArg(4, correlationSurfaceSizeOversampled));
            CL_CHECK_ERROR(fft2dPaddingKernel.setArg(5, correlationSurfaceSizeOversampled));
            queue.enqueueNDRangeKernel(fft2dPaddingKernel, cl::NullRange,
                cl::NDRange(correlationSurfaceSizeOversampled/2, correlationSurfaceSizeOversampled/2));

            // fft back to real space
            CL_CHECK_ERROR(ifft2dKernel.setArg(0, correlationSurfaceOSF));
            CL_CHECK_ERROR(ifft2dKernel.setArg(1, correlationSurfaceOS));
            CL_CHECK_ERROR(ifft2dKernel.setArg(2, correlationSurfaceSizeOversampled));
            CL_CHECK_ERROR(ifft2dKernel.setArg(3, correlationSurfaceSizeOversampled));
            queue.enqueueNDRangeKernel(ifft2dKernel, cl::NullRange,
                cl::NDRange(correlationSurfaceSizeOversampled, correlationSurfaceSizeOversampled));

            // buffer_print<cl_float2>(queue, correlationSurfaceOS,
            //    correlationSurfaceSizeOversampled, correlationSurfaceSizeOversampled,
            //    "correlationSurface OverSampled");

            // find the max location in correlation surface
            groupSize = std::min(
                next_power_of_2(correlationSurfaceSizeOversampled*correlationSurfaceSizeOversampled),
                maxWorkGroupSize);
            CL_CHECK_ERROR(findMaxLocationKernel.setArg(0, correlationSurfaceOS));
            CL_CHECK_ERROR(findMaxLocationKernel.setArg(1, corrSurfaceMaxLoc));
            CL_CHECK_ERROR(findMaxLocationKernel.setArg(2, cl::Local(groupSize*sizeof(float))));
            CL_CHECK_ERROR(findMaxLocationKernel.setArg(3, cl::Local(groupSize*sizeof(int))));
            CL_CHECK_ERROR(findMaxLocationKernel.setArg(4, correlationSurfaceSizeOversampled));
            CL_CHECK_ERROR(findMaxLocationKernel.setArg(5, correlationSurfaceSizeOversampled));
            CL_CHECK_ERROR(findMaxLocationKernel.setArg(6, correlationSurfaceSizeOversampled));
            CL_CHECK_ERROR(findMaxLocationKernel.setArg(7, correlationSurfaceSizeOversampled));
            CL_CHECK_ERROR(queue.enqueueNDRangeKernel(findMaxLocationKernel, cl::NullRange,
                cl::NDRange(groupSize), // globalSize
                cl::NDRange(groupSize)));  // local/Workgroup Size

            // copy max location
            CL_CHECK_ERROR(queue.enqueueReadBuffer(
                corrSurfaceMaxLoc,
                CL_TRUE, // blocking
                0, // offset
                sizeof(cl_int2),
                &offsetFrac));
            // std::cout << "max location second pass " << offsetFrac << "\n";
            // std::cout << "half search " << make_int2(halfSearchRangeAcrossRaw, halfSearchRangeDownRaw) << "\n";

            const int offset_index = iWindowDown*numberWindowAcross+iWindowAcross;
            offset_image[offset_index].x = offsetRaw.x  - halfSearchRangeAcrossRaw
              + (float)offsetFrac.x/(float)oversamplingFactor;
            offset_image[offset_index].y = offsetRaw.y  - halfSearchRangeDownRaw
              + (float)offsetFrac.y/(float)oversamplingFactor;

            // std::cout << "offset " << offset_image[offset_index] << "\n";

        } // end of Across Windows Loop
    } // end of Down Windows Loop

    // write the offset to file
    std::ofstream offsetFile(offsetImageName, std::ios::binary);
    if (!offsetFile) {
        std::cerr << "Failed to open the file for writing." << std::endl;
        exit(EXIT_FAILURE);
    }
    offsetFile.write(reinterpret_cast<const char *>(offset_image),
        cfloatBytes*numberWindowAcross*numberWindowDown);

    std::cout << "The offset image of size " << make_int2(numberWindowAcross, numberWindowDown)
        << " is saved in " << offsetImageName
        << " in BIP - CFLOAT Format (offset_range, offset_azimuth) \n";

    // close all files
    offsetFile.close();
    referenceFile.close();
    secondaryFile.close();

    // clean cl::Buffer
    // all done
}
// end of file
