# Qualcomm Snapdragon Adreno GPU Development Notes

## Platforms

### Snapdragon™ 855 Mobile HDK (HDK 8150) - currently tested
#### Specifications
 - Qualcomm® Kryo™ 485 CPU (ARMv8.2-A)
   - One high-performance Gold Prime core targeting 2.7+ GHz
   - Three high-performance Gold cores targeting 2.4+ GHz
   - Four low-power Kryo cores targeting 1.7 GHz
 - Qualcomm® Adreno™ 640 GPU
   - 768 [384] FP32 ALUs  
   - 585-675 MHz
   - 898.5-1036.8 FP32 GFLOPS
   - API Support: OpenGL® ES 3.2, OpenCL™ 2.0 FP, Vulkan® 1.1
 - Qualcomm® Hexagon™ 690 DSP
 - 6GB LPDDR4x RAM 4266 Quad-channel 16-bit (64-bit) @ 2133 MHz (34.1 GB/s)

#### OpenCL Profile
- Device: QUALCOMM Adreno(TM)
- Vendor: QUALCOMM
- OpenCL Version: OpenCL 2.0 Adreno(TM) 640
- Max Compute Units: 2
- Max Work Group Size: 1024
- Local Memory Size: 32 KB
- Global Memory Size: 2752 MB

### Snapdragon™ 8 Gen 1 Mobile Hardware Development Kit (HDK 8450)
 - Qualcomm® Kryo™ 780 CPU (ARMv9)
 - Qualcomm® Adreno™ 730 GPU
   - 1536 [768] FP32 ALUs  
   - 818-900 MHz
   - 2512.8-2764.8 FP32 GFLOPS
   - API Support: OpenGL® ES 3.2, OpenCL™ 2.0 FP, Vulkan® 1.1
 - 7th-generation Qualcomm AI Engine with Qualcomm® Hexagon™ digital signal processor
 - Qualcomm Spectra™ 680 image signal processor
 - 12GB LPDDR5-3200 Single-channel 128-bit @ 933 MHz (64 GB/s)

### Snapdragon™ 8 Gen 2 Mobile Hardware Development Kit (HDK 8550) - coming 2024
 - Qualcomm® Kryo™ (ARMv9)
   - 64-bit Architecture
   - 1 Prime core, up to 3.36 GHz 
   - 4 Performance cores, up to 2.8 GHz
   - 3 Efficiency cores, up to 2.0 GHz 
 - Qualcomm® Adreno™ 740 GPU
   - 2560 [1280] FP32 ALUs  
   - 680-719 MHz
   - 3481.6-3681.2 FP32 GFLOPS
   - API Support: OpenGL® ES 3.2, OpenCL™ 2.0 FP, Vulkan® 1.3
 - 12GB LPDDR5-3200 Single-channel 128-bit @ 933 MHz (64 GB/s)



