#include "clAmpcor.h"

#include "clProgram.h"
#include "clFFT2d.h"
#include "clCorrelator.h"
#include "clOversampler.h"

#include <iostream>
#include <fstream>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

void cl::Ampcor::Ampcor::read_parameters_from_json(const std::string& filename)
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

        // set the secondary window size accordingly
        secondaryWindowWidthRaw = windowWidthRaw + halfSearchRangeAcrossRaw*2;
        secondaryWindowHeightRaw = windowHeightRaw + halfSearchRangeDownRaw*2;

        skipSampleAcross = settings.at("skip_between_windows").value("across", 128);
        skipSampleDown = settings.at("skip_between_windows").value("down", 128);

        secondaryStartPixelAcross = settings.at("start_pixel_secondary").value("across", secondaryWindowWidthRaw/2);
        secondaryStartPixelDown = settings.at("start_pixel_secondary").value("down", secondaryWindowHeightRaw/2);

        secondaryEndPixelAcross = settings.at("end_pixel_secondary").value("across",
            secondaryImageWidth-secondaryWindowWidthRaw/2);
        secondaryEndPixelDown = settings.at("end_pixel_secondary").value("down",
            secondaryImageHeight-secondaryWindowHeightRaw/2);

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
        secondaryWindowWidth = secondaryWindowWidthRaw*rawDataOversamplingFactor;
        secondaryWindowHeight = secondaryWindowHeightRaw*rawDataOversamplingFactor;

        correlationSurfaceWidth = secondaryWindowWidth - windowWidth + 1;
        correlationSurfaceHeight = secondaryWindowHeight - windowHeight + 1;

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
                << make_int2(secondaryWindowWidth, secondaryWindowHeight) << "\n"
            << "starting pixel (center of the first window)"
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

void cl::Ampcor::Ampcor::run()
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
    // initialize the opencl handles
    clHandle handle(CL_DEVICE_TYPE_GPU);
    cl::Context& context = handle.context;
    cl::Device& device = handle.device;
    // build the kernel program
    handle.program = cl::Ampcor::Program(context);
    cl::Program& program = handle.program;
    cl::CommandQueue queue(context, device);
    //CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE TBD - CL_QUEUE_PROPERTIES

    // ******* CPU/host Buffers *************
    // Create read buffers for reference/secondary images
    size_type referenceBufferSize = referenceImageWidth*windowHeightRaw*cfloatBytes;
    char * referenceBufferHost = new char[referenceBufferSize];
    // Create read buffers for secondary images
    size_type secondaryBufferSize = secondaryImageWidth*secondaryWindowHeightRaw*cfloatBytes;
    char * secondaryBufferHost = new char[secondaryBufferSize];

    // offset image
    cl_float2* offset_image = new cl_float2[numberWindowAcross*numberWindowDown];
    cl_int2 offsetRaw, offsetFrac;

    // ******** GPU/device Buffers ***************
    // for fft, we need to pad zero to a size in power of 2
    int windowWidthP2 = next_power_of_2(secondaryWindowWidth);
    int windowHeightP2 = next_power_of_2(secondaryWindowHeight);

    // reference image (windowWidth, windowHeight), but enlarged to the secondary window size
    cl::Buffer referenceWindow(context, CL_MEM_READ_WRITE,
        windowWidthP2*windowHeightP2*cfloatBytes);
    // secondary image, window + secondary range
    cl::Buffer secondaryWindow(context, CL_MEM_READ_WRITE,
        windowWidthP2*windowHeightP2*cfloatBytes);

    // reference image sum and sum square
    cl::Buffer referenceWindowSum2(context, CL_MEM_READ_WRITE,
        cfloatBytes);
    // secondary image sum area table
    cl::Buffer secondaryWindowSAT2(context, CL_MEM_READ_WRITE,
        secondaryWindowHeight*secondaryWindowWidth*cfloatBytes);

    // correlation surfaces
    cl::Buffer correlationSurface(context, CL_MEM_READ_WRITE,
        windowWidthP2*windowHeightP2*cfloatBytes);
    cl::Buffer correlationSurfaceZoom(context, CL_MEM_READ_WRITE,
        zoomWindowSize*zoomWindowSize*cfloatBytes);
    cl::Buffer correlationSurfaceOS(context, CL_MEM_READ_WRITE,
        correlationSurfaceSizeOversampled*correlationSurfaceSizeOversampled*cfloatBytes);

    // correlation surface max location/offset
    cl::Buffer corrSurfaceMaxLoc(context, CL_MEM_READ_WRITE,
        sizeof(cl_int2));

    // get kernels from the program
    // kernel to take amplitude values for reference window
    cl::Kernel referenceAmplitudeKernel;
    CL_CHECK_ERROR(referenceAmplitudeKernel= cl::Kernel(program, "matrix_complex_amplitude"));
    CL_CHECK_ERROR(referenceAmplitudeKernel.setArg(0, referenceWindow));
    CL_CHECK_ERROR(referenceAmplitudeKernel.setArg(1, windowWidth));
    CL_CHECK_ERROR(referenceAmplitudeKernel.setArg(2, windowHeight));
    CL_CHECK_ERROR(referenceAmplitudeKernel.setArg(3, windowWidthP2));
    CL_CHECK_ERROR(referenceAmplitudeKernel.setArg(4, windowHeightP2));
    cl::NDRange referenceAmplitudeKernel_globalSize(windowWidthP2, windowHeightP2);

    // kernel to take amplitude values for reference window
    cl::Kernel secondaryAmplitudeKernel;
    CL_CHECK_ERROR(secondaryAmplitudeKernel= cl::Kernel(program, "matrix_complex_amplitude"));
    CL_CHECK_ERROR(secondaryAmplitudeKernel.setArg(0, secondaryWindow));
    CL_CHECK_ERROR(secondaryAmplitudeKernel.setArg(1, secondaryWindowWidth));
    CL_CHECK_ERROR(secondaryAmplitudeKernel.setArg(2, secondaryWindowHeight));
    CL_CHECK_ERROR(secondaryAmplitudeKernel.setArg(3, windowWidthP2));
    CL_CHECK_ERROR(secondaryAmplitudeKernel.setArg(4, windowHeightP2));
    cl::NDRange secondaryAmplitudeKernel_globalSize(windowWidthP2, windowHeightP2);

    // kernel to compute sum and sum square of the reference window
    cl::Kernel referenceSumKernel;
    CL_CHECK_ERROR(referenceSumKernel=cl::Kernel(program, "matrix_sum_sum2"));
    size_type maxWorkGroupSize;
    CL_CHECK_ERROR(referenceSumKernel.getWorkGroupInfo(device, CL_KERNEL_WORK_GROUP_SIZE, &maxWorkGroupSize));
    maxWorkGroupSize = std::min(next_power_of_2(windowHeight*windowWidth), maxWorkGroupSize);
    CL_CHECK_ERROR(referenceSumKernel.setArg(0, referenceWindow));
    CL_CHECK_ERROR(referenceSumKernel.setArg(1, referenceWindowSum2));
    CL_CHECK_ERROR(referenceSumKernel.setArg(2, cl::Local(cfloatBytes*maxWorkGroupSize)));
    CL_CHECK_ERROR(referenceSumKernel.setArg(3, windowWidth));
    CL_CHECK_ERROR(referenceSumKernel.setArg(4, windowHeight));
    CL_CHECK_ERROR(referenceSumKernel.setArg(5, windowWidthP2));
    CL_CHECK_ERROR(referenceSumKernel.setArg(6, windowHeightP2));

    cl::NDRange referenceSumKernel_globalSize(maxWorkGroupSize);
    cl::NDRange referenceSumKernel_localSize(maxWorkGroupSize);

    // kernel to compute sum (and sum sq) area table for the secondary window
    cl::Kernel secondarySatKernel;
    CL_CHECK_ERROR(secondarySatKernel=cl::Kernel(program, "matrix_sat_sat2"));
    CL_CHECK_ERROR(secondarySatKernel.setArg(0, secondaryWindow));
    CL_CHECK_ERROR(secondarySatKernel.setArg(1, secondaryWindowSAT2));
    CL_CHECK_ERROR(secondarySatKernel.setArg(2, secondaryWindowWidth));
    CL_CHECK_ERROR(secondarySatKernel.setArg(3, secondaryWindowHeight));
    CL_CHECK_ERROR(secondarySatKernel.setArg(4, windowWidthP2));
    CL_CHECK_ERROR(secondarySatKernel.setArg(5, windowHeightP2));
    CL_CHECK_ERROR(secondarySatKernel.getWorkGroupInfo(device, CL_KERNEL_WORK_GROUP_SIZE, &maxWorkGroupSize));
    maxWorkGroupSize = std::min(static_cast<size_type>(std::max(secondaryWindowWidth, secondaryWindowHeight)),
        maxWorkGroupSize);
    cl::NDRange secondarySatKernel_globalSize(maxWorkGroupSize);
    cl::NDRange secondarySatKernel_localSize(maxWorkGroupSize);

    // cross-correlation (un-normalized) processor
    cl::Ampcor::Correlator correlator(handle,
        windowWidthP2, windowHeightP2,
        referenceWindow,
        secondaryWindow,
        correlationSurface);

    // kernel for normalization
    cl::Kernel corrNormalizeKernel;
    CL_CHECK_ERROR(corrNormalizeKernel=cl::Kernel(program, "correlation_normalize"));
    CL_CHECK_ERROR(corrNormalizeKernel.setArg(0, correlationSurface));
    CL_CHECK_ERROR(corrNormalizeKernel.setArg(1, referenceWindowSum2));
    CL_CHECK_ERROR(corrNormalizeKernel.setArg(2, secondaryWindowSAT2));
    CL_CHECK_ERROR(corrNormalizeKernel.setArg(3, correlationSurfaceWidth));
    CL_CHECK_ERROR(corrNormalizeKernel.setArg(4, correlationSurfaceHeight));
    CL_CHECK_ERROR(corrNormalizeKernel.setArg(5, windowWidthP2));
    CL_CHECK_ERROR(corrNormalizeKernel.setArg(6, windowHeightP2));
    CL_CHECK_ERROR(corrNormalizeKernel.setArg(7, windowWidth));
    CL_CHECK_ERROR(corrNormalizeKernel.setArg(8, windowHeight));
    CL_CHECK_ERROR(corrNormalizeKernel.setArg(9, secondaryWindowWidth));
    CL_CHECK_ERROR(corrNormalizeKernel.setArg(10, secondaryWindowHeight));
    cl::NDRange corrNormalizeKernel_globalSize(correlationSurfaceWidth, correlationSurfaceHeight);

    // kernel for finding the max location in correlation surface
    cl::Kernel findMaxLocationKernel(program, "matrix_max_location");
    CL_CHECK_ERROR(findMaxLocationKernel.getWorkGroupInfo(handle.device, CL_KERNEL_WORK_GROUP_SIZE, &maxWorkGroupSize));
    maxWorkGroupSize = std::min(next_power_of_2(correlationSurfaceWidth*correlationSurfaceHeight),
        maxWorkGroupSize);
    CL_CHECK_ERROR(findMaxLocationKernel.setArg(0, correlationSurface));
    CL_CHECK_ERROR(findMaxLocationKernel.setArg(1, corrSurfaceMaxLoc));
    CL_CHECK_ERROR(findMaxLocationKernel.setArg(2, cl::Local(maxWorkGroupSize*sizeof(float))));
    CL_CHECK_ERROR(findMaxLocationKernel.setArg(3, cl::Local(maxWorkGroupSize*sizeof(int))));
    CL_CHECK_ERROR(findMaxLocationKernel.setArg(4, correlationSurfaceWidth));
    CL_CHECK_ERROR(findMaxLocationKernel.setArg(5, correlationSurfaceHeight));
    CL_CHECK_ERROR(findMaxLocationKernel.setArg(6, windowHeightP2));
    cl::NDRange findMaxLocationKernel_globalSize(maxWorkGroupSize);
    cl::NDRange findMaxLocationKernel_localSize(maxWorkGroupSize);

    // kernel for extracting a small window around the peak position for oversampling
    cl::Kernel extractRealKernel(program, "matrix_extract_real");
    CL_CHECK_ERROR(extractRealKernel.setArg(0, correlationSurface)); // input
    CL_CHECK_ERROR(extractRealKernel.setArg(1, correlationSurfaceZoom)); //output
    CL_CHECK_ERROR(extractRealKernel.setArg(2, correlationSurfaceWidth));  // input actual width
    CL_CHECK_ERROR(extractRealKernel.setArg(3, correlationSurfaceHeight)); // input actual height
    CL_CHECK_ERROR(extractRealKernel.setArg(4, windowWidthP2));  // input storage width / stride
    CL_CHECK_ERROR(extractRealKernel.setArg(5, corrSurfaceMaxLoc)); // extract center
    CL_CHECK_ERROR(extractRealKernel.setArg(6, -halfZoomWindowSizeRaw)); // offset
    CL_CHECK_ERROR(extractRealKernel.setArg(7, -halfZoomWindowSizeRaw)); // offset
    // extract location needs to be updated during the run
    cl::NDRange extractRealKernel_globalSize(zoomWindowSize, zoomWindowSize);

    // oversampler for the correlation surface
    cl::Ampcor::Oversampler correlationOversampler(
        handle, zoomWindowSize, zoomWindowSize,
        correlationSurfaceSizeOversampled, correlationSurfaceSizeOversampled,
        correlationSurfaceZoom, correlationSurfaceOS);

    // kernel for finding the max location in the oversampled correlation surface
    cl::Kernel findMaxLocationOSKernel(program, "matrix_max_location");
    CL_CHECK_ERROR(findMaxLocationOSKernel.getWorkGroupInfo(device, CL_KERNEL_WORK_GROUP_SIZE, &maxWorkGroupSize));
    maxWorkGroupSize = std::min(next_power_of_2(correlationSurfaceSizeOversampled*correlationSurfaceSizeOversampled),
        maxWorkGroupSize);
    CL_CHECK_ERROR(findMaxLocationOSKernel.setArg(0, correlationSurfaceOS));
    CL_CHECK_ERROR(findMaxLocationOSKernel.setArg(1, corrSurfaceMaxLoc));
    CL_CHECK_ERROR(findMaxLocationOSKernel.setArg(2, cl::Local(maxWorkGroupSize*sizeof(float))));
    CL_CHECK_ERROR(findMaxLocationOSKernel.setArg(3, cl::Local(maxWorkGroupSize*sizeof(int))));
    CL_CHECK_ERROR(findMaxLocationOSKernel.setArg(4, correlationSurfaceSizeOversampled));
    CL_CHECK_ERROR(findMaxLocationOSKernel.setArg(5, correlationSurfaceSizeOversampled));
    CL_CHECK_ERROR(findMaxLocationOSKernel.setArg(6, correlationSurfaceSizeOversampled));
    cl::NDRange findMaxLocationOSKernel_globalSize(maxWorkGroupSize);
    cl::NDRange findMaxLocationOSKernel_localSize(maxWorkGroupSize);


    // ************* Processing ************
    // message interval
    int_type message_interval = std::max(numberWindowDown/10, 1);
    // iterative over windows along height
    for(int_type iWindowDown=0; iWindowDown<numberWindowDown; iWindowDown++)
    {
        // **** read image to buffers
        // determine the starting line(s)
        size_type secondaryLineStart = secondaryStartPixelDown - secondaryWindowHeightRaw/2
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
            size_type secondaryColStart = secondaryStartPixelAcross - secondaryWindowWidthRaw/2
                + iWindowAcross*skipSampleAcross;
            size_type referenceColStart = secondaryColStart + halfSearchRangeAcrossRaw;

            // std::cout << "referenceStart " << referenceColStart << " " << referenceLineStart << "\n";
            // std::cout << "secondaryStart " << secondaryColStart << " " << secondaryLineStart << "\n";

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
            CL_CHECK_ERROR(queue.enqueueWriteBufferRect(
                referenceWindow, // buffer
                CL_TRUE, // blocking
                d_origin, // buffer origin
                s_origin, // host origin
                region,   // rect region
                windowWidthP2*cfloatBytes,       // dst buffer_row_pitch
                0,    // buffer_slice_pitch, n/a for 1d/2d
                referenceImageWidth*cfloatBytes,       // host_row_pitch
                0,    // host_slice_pitch
                referenceBufferHost // host posize_typeer
                ));

            // take amplitude
            CL_CHECK_ERROR(queue.enqueueNDRangeKernel(
                referenceAmplitudeKernel,
                cl::NullRange,
                referenceAmplitudeKernel_globalSize
                ));

#ifdef CL_AMPCOR_STEP_DEBUG
            buffer_debug<cl_float2>(queue, referenceWindow,
                windowWidthP2, windowHeightP2, "reference amplitude");
#endif
            // compute the sum and sum square of reference window - for normalization
            CL_CHECK_ERROR(queue.enqueueNDRangeKernel(
                referenceSumKernel,
                cl::NullRange,
                referenceSumKernel_globalSize, // globalSize
                referenceSumKernel_globalSize
                ));  // local/Workgroup Size

#ifdef CL_AMPCOR_STEP_DEBUG
            buffer_debug<cl_float2>(queue, referenceWindowSum2,
                1, 1, "reference sum");
#endif

            // copy a window from secondary buffer
            s_origin[0] = secondaryColStart*cfloatBytes;
            region[0] = secondaryWindowWidth*cfloatBytes;
            region[1] = secondaryWindowHeight;
            CL_CHECK_ERROR(queue.enqueueWriteBufferRect(
                secondaryWindow, // buffer
                CL_TRUE, // blocking
                d_origin, // buffer origin
                s_origin, // host origin
                region,   // rect region
                windowWidthP2*cfloatBytes,       // dst buffer_row_pitch
                0,    // buffer_slice_pitch, n/a for 1d/2d
                secondaryImageWidth*cfloatBytes,       // host_row_pitch
                0,    // host_slice_pitch
                secondaryBufferHost // host posize_typeer
                ));


            // take the amplitude
            CL_CHECK_ERROR(queue.enqueueNDRangeKernel(
                secondaryAmplitudeKernel,
                cl::NullRange,
                secondaryAmplitudeKernel_globalSize //globalSize
                ));

#ifdef CL_AMPCOR_STEP_DEBUG
            buffer_debug<cl_float2>(queue, secondaryWindow,
                windowWidthP2, windowHeightP2, "secondaryWindow amplitude");
#endif

            // compute the sum area table
            CL_CHECK_ERROR(queue.enqueueNDRangeKernel(
                secondarySatKernel,
                cl::NullRange,
                secondarySatKernel_globalSize,
                secondarySatKernel_localSize
                ));  // local/Workgroup Size

#ifdef CL_AMPCOR_STEP_DEBUG
            buffer_debug<cl_float2>(queue, secondaryWindowSAT2,
                secondaryWindowWidth, secondaryWindowHeight, "secondary SAT");
#endif
            // cross-correlation
            correlator.execute(queue);

#ifdef CL_AMPCOR_STEP_DEBUG
            buffer_debug<cl_float2>(queue, correlationSurface,
                windowWidthP2, windowHeightP2, "correlation large");
#endif

            // normalize the correlation surface
            CL_CHECK_ERROR(queue.enqueueNDRangeKernel(
                corrNormalizeKernel,
                cl::NullRange,
                corrNormalizeKernel_globalSize
                ));

#ifdef CL_AMPCOR_STEP_DEBUG
            buffer_debug<cl_float2>(queue, correlationSurface,
                windowWidthP2, windowHeightP2, "correlation normalized");
#endif

            // find the max location in correlation surface
            CL_CHECK_ERROR(queue.enqueueNDRangeKernel(
                findMaxLocationKernel,
                cl::NullRange,
                findMaxLocationKernel_globalSize, // globalSize
                findMaxLocationKernel_localSize
                ));

            // copy max location
            CL_CHECK_ERROR(queue.enqueueReadBuffer(
                corrSurfaceMaxLoc,
                CL_TRUE, // blocking
                0, // offset
                sizeof(cl_int2),
                &offsetRaw
                ));

            offsetRaw.x -= halfZoomWindowSizeRaw;
            offsetRaw.y -= halfZoomWindowSizeRaw;

#ifdef CL_AMPCOR_STEP_DEBUG
            std::cout << "max location first pass " << offsetRaw << "\n";
#endif
            // extract the real part and the top corners to get the correlation surface
            CL_CHECK_ERROR(queue.enqueueNDRangeKernel(
                extractRealKernel,
                cl::NullRange,
                extractRealKernel_globalSize
                ));

#ifdef CL_AMPCOR_STEP_DEBUG
            buffer_debug<cl_float2>(queue, correlationSurfaceZoom,
                zoomWindowSize, zoomWindowSize, "correlationSurfaceZoom");
#endif

            /// use fft to oversample the correlation surface
            correlationOversampler.execute(queue);

#ifdef CL_AMPCOR_STEP_DEBUG
            buffer_debug<cl_float2>(queue, correlationSurfaceOS,
                correlationSurfaceSizeOversampled, correlationSurfaceSizeOversampled,
                "correlationSurface OverSampled");
#endif
            // find the max location in correlation surface
            CL_CHECK_ERROR(queue.enqueueNDRangeKernel(
                findMaxLocationOSKernel,
                cl::NullRange,
                findMaxLocationOSKernel_globalSize, // globalSize
                findMaxLocationOSKernel_localSize
                ));  // local/Workgroup Size

            // copy max location
            CL_CHECK_ERROR(queue.enqueueReadBuffer(
                corrSurfaceMaxLoc,
                CL_TRUE, // blocking
                0, // offset
                sizeof(cl_int2),
                &offsetFrac));

#ifdef CL_AMPCOR_STEP_DEBUG
            std::cout << "max location second pass " << offsetFrac << "\n";
            std::cout << "half secondary " << make_int2(halfSearchRangeAcrossRaw, halfSearchRangeDownRaw) << "\n";
#endif

            const int offset_index = iWindowDown*numberWindowAcross+iWindowAcross;
            offset_image[offset_index].x = offsetRaw.x  - halfSearchRangeAcrossRaw
              + (float)offsetFrac.x/(float)oversamplingFactor;
            offset_image[offset_index].y = offsetRaw.y  - halfSearchRangeDownRaw
              + (float)offsetFrac.y/(float)oversamplingFactor;

#ifdef CL_AMPCOR_STEP_DEBUG
            std::cout << "offset " << offset_image[offset_index] << "\n";
#endif
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
