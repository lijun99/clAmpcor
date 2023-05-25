# clAmpcor - Ampcor with OpenCL 

Ampcor (amplitude cross-correlation) uses feature-tracking method to determine offsets (surface displacments) from a pair of SAR images taken at different times. See ROIPAC or [ISCE](https://github.com/isce-framework/isce2) for more details.  

Many computations in Ampcor can be accelerated by GPUs - multiple arithmetic units run in parallel, including

 - DFT (Discrete Fourier Transform), used for oversampling, and cross-correlation 
 - Parallel Reduction, such as finding max value in a correlation surface, computing summation in correlation surface normalization. 

This OpenCL implementation follows the CUDA version, [PyCuAmpcor](https://github.com/isce-framework/isce2/tree/main/contrib/PyCuAmpcor), while it supports more GPU platforms, NVIDIA, AMD, Apple M1/M2, and mobile devices, such as Adreno GPUs in Qualcomm Snapdragon Mobile processors. 

## Installation 

Please download the source code from this [github repo](https:://lijun99.github.com/clAmpcor), and use cmake to build the application. 

### Linux 

```cmake
    cmake -S . -B build
    make -C build
```
You may find `clAmpcor` as a shell command inside `build` directory. 

### MacOSX

Install Xcode or command line tools. 

Apple only supports OpenCL to 1.2. You may need to copy the OpenCL C++ wrapper to the SDK OpenCL framework, e.g., 

```
    sudo cp include/CL/*.hpp /Library/Developer/CommandLineTools/SDKs/MacOSX13.3.sdk/System/Library/Frameworks/OpenCL.framework/Headers
```
The follow the same instruction as in Linux. 

### Android 

#### Setup Android-SDK
Set up Android SDK, NDK, Platform-tools. 

#### Get the OpenCL library from the device

Grab the libOpenCL.so from the device (pre-built by manufacturers), 

```
    adb pull /vendor/lib64/libOpenCL.so
```
Please it under android/qualcomm/lib (to replace the current file). 

#### Build 
Please modify the `CMakeLists.txt` under `android` directory to specify the correct paths for android sdk and device architectures.  

```commandline
    cd android 
    cmake -S . -B build
    make -C build    
```

If you meet the error: 
```
ld: error: adreno_opencl/deviceQuery/lib/libOpenCL.so: invalid sh_info in symbol table
clang++: error: linker command failed with exit code 1 (use -v to see invocation),
 ```

 follow the instructions below to fix it 
 - visit https://elfy.io/
 - Open/upload libOpenCL.so,
 - Section headers->Elf_Shdr->2-> Edit and Commit the following 0x2c sh_info change to 0x1, or binary 01 00 00 00
 - Save the edited libOpenCL.so file.

#### adb - Upload/download files and run programs

```
    adb push mycommand /data/local/tmp # upload a file to device (non-rooted)
    adb pull /path/to/file_on_device # download a file
    adb root # restart adb with root access
    adb shell # open a shell to run commands on devices
```

If some library is missing when running a program,  use 

```
    export LD_LIBRARY_PATH=/system/vendor/lib64:$LD_LIBRARY_PATH
```

## Usage 

In a work directory, where a pair of SLC (Single Look Complex) imagess are located, create a JSON config file, `ampcor.json` following the [example](examples/ampcor.json). Adjust the settings, and run `clAmpcor`.   



