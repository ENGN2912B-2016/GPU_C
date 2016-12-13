# GPU_C
The repository for Jiacheng Guo, Tianxiong Wang and Tianlun Liu

Summary of project:
	We complete three functions of image processing through GPU progamming,converting into greyscale image,bluring image, and conner detection on image. Each function has its own file. There will be detail instrcution for each function as shown below. 
	We also have a functional user interface, which allows user to load an image, to press buttons for each function, and to output correspondding result image for each function. 


How everything works:

User Interface:

Commonents:Since CUDA and Qt are not compitable, or there is no appropriate way to let them be compitable due to time constraint.Our User interface is just a demo, the result is processed by our function block but not call from the interface.Ideally, it should run by the interface and output the result. Also, result image may not be show up because of the path issue. If you have such problem, you may replace the file path in program with your current file  path to fix the issue.

How it runs:
Open the program, and press "Open" button to load a picture.
Press each function buttons, "Grey", "Blur" and "Corner Detection", the corresponding result will appear in the result frame.

Greyscale:

Princieple: Loading a color image into program,and GPU initializes its threads for image processing. Each GPU thread reads each pixel of the image. Each pixel is composed by 3 channels,and using values in these three channel through the greyscale formula to calculate the greyvalue of each pixel. Finally, reconstrcut each pixel by its grey value to obtain the greyscale image. 

How it runs:


Blurred:

Principle:Loading a color image into program, and GPU initializes its threads for image processing.Each GPU thread reads each pixel of the image,and extracts each channel of a pixel,to reconstruct image for each channel to obatin channel image.Then,original image divided into three image based on channels. Using blur formula to blur each channel image, then combine these three images into one image based on corresponding pixel. The blurred image is generated. 

How it runs:

Conner detection:

Principle:Loading a color image into program, and GPU initializes its threads for image processing.Each GPU thread reads each pixel of the image. Using FAST algortithm to detect corners in image, and using OpenCV to label each corner on the graph. 

How it runs:
