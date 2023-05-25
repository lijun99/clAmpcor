/**
 * @file  parameter.h
 * @brief A class holds Ampcor process parameters
 *
 */

// code guard
#ifndef __CLAMPCOR_H__
#define __CLAMPCOR_H__

#include <string>
#include "clHelper.h"

// wrapped in a namesapce
namespace cl {

struct Ampcor {

    using size_type = cl::size_type;
    using complex_type = cl_float2;
    using float_type = cl_float;
    using int_type = cl_int;

    const size_type cfloatBytes = sizeof(complex_type);
    const size_type floatBytes = sizeof(float_type);

    //reference image
    std::string referenceImageName;    ///< reference SLC image name
    int_type imageDataType1;                ///< reference image data type, 2=cfloat/complex/float2 1=float
    int_type referenceImageHeight;          ///< reference image height
    int_type referenceImageWidth;           ///< reference image width

    //secondary image
    std::string secondaryImageName;     ///< secondary SLC image name
    int_type imageDataType2;                 ///< secondary image data type
    int_type secondaryImageHeight;           ///< secondary image height
    int_type secondaryImageWidth;            ///< secondary image width

    std::string offsetImageName;       ///< Offset fields output filename
    std::string snrImageName;          ///< Output SNR filename
    std::string covImageName;          ///< Output variance filename

    // chip or window size for raw data
    int_type windowHeightRaw;        ///< Template window height (original size)
    int_type windowWidthRaw;         ///< Template window width (original size)
    int_type searchWindowHeightRaw;  ///< Search window height (original size)
    int_type searchWindowWidthRaw;   ///< Search window width (orignal size)

    int_type halfSearchRangeDownRaw;   ///< (searchWindowHeightRaw-windowHeightRaw)/2
    int_type halfSearchRangeAcrossRaw;    ///< (searchWindowWidthRaw-windowWidthRaw)/2
    // search range is (-halfSearchRangeRaw, halfSearchRangeRaw)

    int_type searchWindowHeightRawZoomIn; ///< search window height used for zoom in
    int_type searchWindowWidthRawZoomIn;  ///< search window width used for zoom in

    // chip or window size after oversampling
    const int_type rawDataOversamplingFactor=1;  ///< Raw data overampling factor (from original size to oversampled size)
    int_type windowHeight;           ///< Template window length (oversampled size)
    int_type windowWidth;            ///< Template window width (original size)
    int_type searchWindowHeight;     ///< Search window height (oversampled size)
    int_type searchWindowWidth;      ///< Search window width (oversampled size)

    // strides between chips/windows
    int_type skipSampleDown;   ///< Skip size between neighboring windows in Down direction (original size)
    int_type skipSampleAcross; ///< Skip size between neighboring windows in across direction (original size)

    // correlation surface size
    int_type corrStatWindowSize;     ///< correlation surface size used to estimate snr
    int_type correlationSurfaceWidth; ///< correlation surface size for raw data xcor
    int_type correlationSurfaceHeight;

    // Zoom in region near location of max correlation
    int_type zoomWindowSize;      ///< Zoom-in window size in correlation surface (same for down and across directions)
    int_type halfZoomWindowSizeRaw; ///<  half of zoomWindowSize/rawDataOversamplingFactor
    int_type correlationSurfaceSizeOversampled; /// width and height for oversampled correlation surface

    int_type oversamplingFactor;  ///< Oversampling factor for int_typeerpolating correlation surface

    float thresholdSNR;      ///< Threshold of Signal noise ratio to remove noisy data

    // total number of chips/windows
    int_type numberWindowDown;           ///< number of total windows (down)
    int_type numberWindowAcross;         ///< number of total windows (across)
    int_type numberWindows; 				///< numberWindowDown*numberWindowAcross

    int_type secondaryStartPixelDown;    ///< first starting pixel(used as center) in reference image (down)
    int_type secondaryStartPixelAcross;  ///< first starting pixel(used as center) in reference image (across)
    int_type secondaryEndPixelDown;    ///< first starting pixel in reference image (down)
    int_type secondaryEndPixelAcross;  ///< first starting pixel in reference image (across)

    // methods
    void read_parameters_from_json(const std::string& filename);
    void run();

};

} // end of namespace
#endif //__CLAMPCOR_PARAMETERS_H__
 // end of file
