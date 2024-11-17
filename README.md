# Connected Components (with CUDA)
The connected components problem is common in image processing. FolkeV's [repository](https://github.com/FolkeV/CUDA_CCL) is an extension of original [Playne](https://github.com/DanielPlayne/playne-equivalence-algorithm)'s code with the Playne-Equivalence Connected-Component Labelling Algorithm described in:

D. P. Playne and K. Hawick,<br/>
"A New Algorithm for Parallel Connected-Component Labelling on GPUs,"<br/>
in IEEE Transactions on Parallel and Distributed Systems,<br/>
vol. 29, no. 6, pp. 1217-1230, 1 June 2018.<br/>
* URL: https://ieeexplore.ieee.org/document/8274991

<img src="https://img.shields.io/badge/cuda-30%25-red">&nbsp;<img src="https://img.shields.io/badge/C%2FC%2B%2B-100%25-green">&nbsp;<img src="https://img.shields.io/badge/CMakelists.txt-tested-blue">&nbsp;<img src="https://img.shields.io/badge/Ubuntu-20.4-pink">&nbsp;<img src="https://img.shields.io/badge/Ubuntu-22.4-gold">


Here, the FolkeV's code was download in the CUDA_CCL folder to be used as library with some other features (area and coordinates extraction) very common in OpenCV C++ libraries but not available in cv::cuda classes to be exectued on the GPU.

![doc](doc.gif)

## Usage

- [x] Declare ```cudalabel``` class object:
```
cudalabel labels;
```
- [ ] Set image from CPU:
```
labels.setimg(image);
```
- [x] Or, alternatively, set from GPU to avoid internal uploads, depending on your code needs:
```
labels.setgpuimg(gpuimage);
```
- [x] Step (2), preprocess (uses FolkeV's code in the CPU version).
```
labels.preprocess();
```
- [x] Step (3), internal call to the Playne&Hawick (2018) cuda code.
```
labels.labelize();
```
- [x] Step (4), new functions, get the labels coordinates (internal call to kernel ```kgetinfo```):
```
unsigned int** labelinfo;
labelinfo = labels.getinfo();
```
- [x] Release resources (required if work continues).
```
labels.reset();
```
- [ ] Optional (generate image and save results):
```
if (labels.imgen())
    labels.lsave(output_name);  
```

## Struct labels

The output ```unsigned int **infolables**``` as a double pointer, can be show as a 2D array where first column [0] has the id of the bounding box and positions [1-4] store the corner of the bounding box positions.

| labels |     id | x(0) | x(n) | y(0) | y(n) |
|--------|--------|------|------|------|------|
|      0 |  86758 |  741 |  873 |   84 |  216 |
|      1 | 133306 |  185 |  375 |  130 |  300 |


## Compilation
Download code and call ```cmake```.
```
$ git clone https://github.com/armengot/cudalabel
$ cd cudalabel
$ mkdir build
$ cd build
$ cmake ..
$ make
```

## CPU/GPU times
Using ```timing``` executable tool.
Using the CPU image.
```
[CPU] processing image [2880x2160] with ../samples/sample0.jpg took 59 milliseconds.
[CPU] processing image [2880x2160] with ../samples/sample1.jpg took 61 milliseconds.
[CPU] processing image [2880x2160] with ../samples/sample2.jpg took 61 milliseconds.
[CPU] processing image [2880x2160] with ../samples/sample3.jpg took 61 milliseconds.
[CPU] processing image [2880x2160] with ../samples/sample4.jpg took 60 milliseconds.
[CPU] processing image [2880x2160] with ../samples/sample5.jpg took 62 milliseconds.
[CPU] processing image [2880x2160] with ../samples/sample6.jpg took 62 milliseconds.
[CPU] processing image [2880x2160] with ../samples/sample7.jpg took 62 milliseconds.
[CPU] processing image [2880x2160] with ../samples/sample8.jpg took 61 milliseconds.
[CPU] processing image [2880x2160] with ../samples/sample9.jpg took 62 milliseconds.
```
Uploading the image to the GPU.
```
[GPU] processing image [2880x2160] with ../samples/sample0.jpg took 129 milliseconds.
[GPU] processing image [2880x2160] with ../samples/sample1.jpg took 46  milliseconds.
[GPU] processing image [2880x2160] with ../samples/sample2.jpg took 45  milliseconds.
[GPU] processing image [2880x2160] with ../samples/sample3.jpg took 46  milliseconds.
[GPU] processing image [2880x2160] with ../samples/sample4.jpg took 47  milliseconds.
[GPU] processing image [2880x2160] with ../samples/sample5.jpg took 47  milliseconds.
[GPU] processing image [2880x2160] with ../samples/sample6.jpg took 51  milliseconds.
[GPU] processing image [2880x2160] with ../samples/sample7.jpg took 48  milliseconds.
[GPU] processing image [2880x2160] with ../samples/sample8.jpg took 47  milliseconds.
[GPU] processing image [2880x2160] with ../samples/sample9.jpg took 49  milliseconds.
```

