# Benchmark Results

With settings in examples/ampcor.json, number of windows (152, 430), and the time reported by 

```commandline
    time ./clAmpcor
```

## Qualcomm Adreno 640 (Snapdragon 855 SOC)

- Device: QUALCOMM Adreno(TM)
- Vendor: QUALCOMM
- OpenCL Version: OpenCL 2.0 Adreno(TM) 640
- Max Compute Units: 2 (2x384 = 768 ALUs)
- Max Work Group Size: 1024
- Local Memory Size: 32 KB
- Global Memory Size: 2752 MB
- Processing Power: FP32 898.5-1036.8 GFLOPS
- OpenCL Extensions: cl_khr_3d_image_writes cl_img_egl_image cl_khr_byte_addressable_store cl_khr_depth_images cl_khr_egl_event cl_khr_egl_image cl_khr_fp16 cl_khr_gl_sharing cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics cl_khr_local_int32_base_atomics cl_khr_local_int32_extended_atomics cl_khr_image2d_from_buffer cl_khr_mipmap_image cl_khr_srgb_image_writes cl_khr_subgroups cl_qcom_create_buffer_from_image cl_qcom_ext_host_ptr cl_qcom_ion_host_ptr cl_qcom_perf_hint cl_qcom_other_image cl_qcom_subgroup_shuffle cl_qcom_vector_image_ops cl_qcom_extract_image_plane cl_qcom_android_native_buffer_host_ptr cl_qcom_protected_context cl_qcom_priority_hint cl_qcom_compressed_yuv_image_read cl_qcom_compressed_image cl_qcom_ext_host_ptr_iocoherent cl_qcom_accelerated_image_ops 

### processing time 
- version 1
  - 78m45.22s real     
  - 2m40.39s user    
  - 11m13.75s system
- version 2
  - 8m28.30s real 
  - 2m09.09s user
  - 6m27.47s system 


## NVIDIA GeForce RTX 3060 

- Device: NVIDIA GeForce RTX 3060
- Vendor: NVIDIA Corporation
- OpenCL Version: OpenCL 3.0 CUDA
- Max Compute Units: 28 (28x128 = 3584 ALUs)
- Max Work Group Size: 1024
- Local Memory Size: 48 KB
- Global Memory Size: 12035 MB
- Processing Power: FP32 9.46-12.74 TFLOPS
- OpenCL Extensions: cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics cl_khr_local_int32_base_atomics cl_khr_local_int32_extended_atomics cl_khr_fp64 cl_khr_3d_image_writes cl_khr_byte_addressable_store cl_khr_icd cl_khr_gl_sharing cl_nv_compiler_options cl_nv_device_attribute_query cl_nv_pragma_unroll cl_nv_copy_opts cl_nv_create_buffer cl_khr_int64_base_atomics cl_khr_int64_extended_atomics cl_khr_device_uuid cl_khr_pci_bus_info cl_khr_external_semaphore cl_khr_external_memory cl_khr_external_semaphore_opaque_fd cl_khr_external_memory_opaque_fd

### processing time
- version 1
  - real	3m43.539s
  - user	3m27.562s
  - sys	0m9.144s
- version 2
  - real	0m42.594s
  - user	0m17.572s
  - sys	0m8.899s

  
## Apple Mac Airbook M1

- Device: Apple M1
- Vendor: Apple
- OpenCL Version: OpenCL 1.2
- Max Compute Units: 8 (8x128 =1024 ALUs)
- Max Work Group Size: 256
- Local Memory Size: 32 KB
- Global Memory Size: 5461 M
- Processing Power: FP32 2.6 TFLOPS (theoretical)

### processing time 
- version 1
  - 56.47s user 
  - 188.86s system 
  - 9% cpu 
  - 44:03.58 total
- version 2
  - 36.89s user 
  - 140.12s system 
  - 69% cpu 
  - 4:13.06 total
