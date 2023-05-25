# Benchmark Results

With settings in examples/ampcor.json, number of windows (152, 430), and the time reported by 

```commandline
    time ./clAmpcor
```


## NVIDIA GeForce RTX 3060 

Device: NVIDIA GeForce RTX 3060
Vendor: NVIDIA Corporation
OpenCL Version: OpenCL 3.0 CUDA
Max Compute Units: 28 (28x128 = 3584 ALUs)
Max Work Group Size: 1024
Local Memory Size: 48 KB
Global Memory Size: 12035 MB
Processing Power: FP32 9.46-12.74 TFLOPS

### processing time
real	3m43.539s
user	3m27.562s
sys	0m9.144s

## Qualcomm Adreno 640 (Snapdragon 855 SOC)

Device: QUALCOMM Adreno(TM)
Vendor: QUALCOMM
OpenCL Version: OpenCL 2.0 Adreno(TM) 640
Max Compute Units: 2 (2x384 = 768 ALUs)
Max Work Group Size: 1024
Local Memory Size: 32 KB
Global Memory Size: 2752 MB
Processing Power: FP32 898.5-1036.8 GFLOPS

### processing time 
78m45.22s real     
2m40.39s user    
11m13.75s system

## Apple Mac Airbook M1

Device: Apple M1
Vendor: Apple
OpenCL Version: OpenCL 1.2
Max Compute Units: 8 (8x128 =1024 ALUs)
Max Work Group Size: 256
Local Memory Size: 32 KB
Global Memory Size: 5461 M
Processing Power: FP32 2.6 TFLOPS (theoretical)

### processing time 

56.47s user 188.86s system 9% cpu 44:03.58 total




