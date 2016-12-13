# GPU_C
The repository for Jiacheng Guo, Tianxiong Wang and Tianlun Liu

Summary of project:
	We complete three functions of image processing through GPU progamming,converting into greyscale image,bluring image, and conner detection on image. Each function has its own file. There will be detail instrcution for each function as shown below. 
	We also have a functional user interface, which allows user to load an image, to press buttons for each function, and to output correspondding result image for each function. 


How everything works:
Instructions for runing the code locally:

In order to run these codes, you would have to install CUDA developed by Nvidia and OpenCV.

As for CUDA, We would recommend you installing CUDA Toolkit 8.0, which contains the NVIDIA CUDA C/C++ Compiler(NVCC) for compiling and Nsight Integrated Development Environments for debugging.
You can find it at: https://developer.nvidia.com/cuda-toolkit

The specific installation guides for different operation systems can be found at:
http://docs.nvidia.com/cuda/index.html#axzz4SVbtQAvS

As for OpenCV, the version we used is 2.4.13.1, however we think the lower version would also work. If you use Mac to install OpenCV, we would suggest you installing it by Homebrew, which is:
run "brew install opencv" in terminal.

When loading the codes into the Nsight, you would need to have some modifications:
In Project -> Properties, NVCC Compiler needs to include path "/usr/local/Cellar/opencv/2.4.13.1/include", NVCC Linker needs to include path "/usr/local/Cellar/opencv/2.4.13.1/lib" and include library "opencv_core opencv_imgproc opencv_highgui". These paths might needs to be changed according to your system settings.

There is a Makefile in each subdirectory, which is generated automatically by Nsight, so I'm not sure if it could work on your machine with this Makefile. I would suggest loading these codes by Nsight, which is how we did to run the code.

For each code, the result would be two images, one is the original and the other one is the processed image. You might need to change the path to specific image in the code.

User Interface:

Commonents:Since CUDA and Qt are not compitable, or there is no appropriate way to let them be compitable due to time constraint.Our User interface is just a demo, the result is processed by our function block but not call from the interface.Ideally, it should run by the interface and output the result. Also, result image may not be show up because of the path issue. If you have such problem, you may replace the file path in program with your current file  path to fix the issue.

Open the program, and press "Open" button to load a picture.
Press each function buttons, "Grey", "Blur" and "Corner Detection", the corresponding result will appear in the result frame.

Greyscale:

Princieple: Loading a color image into program,and GPU initializes its threads for image processing. Each GPU thread reads each pixel of the image. Each pixel is composed by 3 channels,and using values in these three channel through the greyscale formula to calculate the greyvalue of each pixel. Finally, reconstrcut each pixel by its grey value to obtain the greyscale image. 

Blurred:

Principle:Loading a color image into program, and GPU initializes its threads for image processing.Each GPU thread reads each pixel of the image,and extracts each channel of a pixel,to reconstruct image for each channel to obatin channel image.Then,original image divided into three image based on channels. Using blur formula to blur each channel image, then combine these three images into one image based on corresponding pixel. The blurred image is generated. 


Conner detection:

Principle:Loading a color image into program, and GPU initializes its threads for image processing.Each GPU thread reads each pixel of the image. Using FAST algortithm to detect corners in image, and using OpenCV to label each corner on the graph. 

