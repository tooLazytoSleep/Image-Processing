# Image-Processing
Many Kinds method to process images or videos
For procesing1：
    ‘i’: reload the original image.
    ‘w’: save the current image into the file ‘ouput.jpg.’
    ‘g’ : convert the image to grayscale using the OpenCV conversion function.
    ‘G’: convert the image to grayscale using your implementation of conversion function.
    ‘c’: cycle through the color channels of the image showing a different channel every time the key is pressed.
    ‘s’: convert the image to grayscale and smooth it using the openCV function. Use a track bar to control the amount of smoothing.
    ‘S’: convert the image to grayscale and smooth it using your function which should perform convolution with a suitable filter. Use a track bar to control the amount of smoothing.
    ‘d’: downsample the image by a factor of 2 without smooting.
    ‘D’: downsample the image by a factor of 2 with smoothing.
    ‘x’: convert the image grayscale and perform convolution with an x derivative filter. Normalize the obtained values to the range [0,255].
    ‘y’: convert the image to grayscale and perform convolution with a y derivative filter. Normalize the obtained values to the range [0,255].
    ‘m’: show the magnitude of the gradient normalized to the range [0,255]. The gradient is computed base don the x and y derivatives of the image.
    ‘p’: convert the image to grayscale and plot the gradient vectors of the image every N pixel and let the plotted gradient vectors have a lenght of K. Use a track bar to control N. Plot the vectors as short line segments of length K.
    ‘r’: convert the image to grayscale and rotate it using an angle of teta degrees. Use a track bar to control the rotation angle.
    ‘h’: display a short description of the program, its command line arguments, and the keys it supports.
For processing2:
1. Detect edge pixels so that you have a binary edge image.
2. Apply the Hough transform to detect straight lines.
3. Determine the edge pixels belonging to each detected line and refine the line parameters estimate using least squares error fitting.
4. Using color draw detected line segments and color the pixels belonging to them.
5. Interactively controlled parameters should include: an edge detection threshold parameter controlling the number of edge pixels detected, the bin size in the parameter (Hough) plane, and the peak detection threshold.
6. Add a mode for displaying the results before and after the least squares refinement overlaid in color over the original (grayscale) image or the binary edge image.
7. Add a mode for displaying the parameter plane (normalized to [0, 255]).
8. The main parameters of each algorithm should be made available for interactive manipulation through keyboard/ mouse/ trackbar interaction.

For extract features:
1.Write a program to extract feature points from the calibration target and show them on the image. Alternatively write a program that allows you to interactively mark the points on the image. Save the image points detected and/or manually entered in a file.
2.Write a second program to compute camera parameters. The non-planar calibration is to name of a single point correspondence file. A point correspondence file is a text file containing in each row a pair of corresponding points(3D-2D) as real number s separated by space. The first 3 numbers in each row give the x, y, z coordinates of the corresponding 2D image point. After completing the calibration, the program should display the intrinsic and extrinsic parameter of the camera as determined by the calibration process. The program should compute and display the mean square error between the known and computed position of the image points. The computed position should be obtained by using the estimated camera parameters to project the 3D points on to the image pane.
3.Implement the RANSAC algorithm for robust estimation.
