# Connected Components (with CUDA)
The connected components problem is common in image processing. FolkeV's [repository](https://github.com/FolkeV/CUDA_CCL) is an extension of original [Playne](https://github.com/DanielPlayne/playne-equivalence-algorithm)'s code with the Playne-Equivalence Connected-Component Labelling Algorithm described in:

D. P. Playne and K. Hawick,<br/>
"A New Algorithm for Parallel Connected-Component Labelling on GPUs,"<br/>
in IEEE Transactions on Parallel and Distributed Systems,<br/>
vol. 29, no. 6, pp. 1217-1230, 1 June 2018.<br/>
* URL: https://ieeexplore.ieee.org/document/8274991

Here, the FolkeV's code was download in the CUDA_CCL folder to be used as library with some other features (area and coordinates extraction) very common in OpenCV C++ libraries but not available in cv::cuda classes to be exectued on the GPU.
