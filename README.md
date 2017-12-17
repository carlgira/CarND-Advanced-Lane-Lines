## Advanced Lane Finding Project

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/chess_boards.png "Chess boards"
[image2]: ./examples/hls_cs.png "HLS Color Space"
[image3]: ./examples/hsv_cs.png "HSV Color Space"
[image4]: ./examples/rgb_cs.png "RGB Color Space"
[image5]: ./examples/sobel_transformation.png "Sobel transformations"
[image6]: ./examples/perspective_transform.png "Perspective transform"
[image7]: ./examples/lane_on_orig.png "Draw Lanes"
[image8]: ./examples/undistorted_img1.png "Undistorted Image 1"
[image9]: ./examples/undistorted_img2.png "Undistorted Image 2"
[image10]: ./examples/curvature_img.png "Curvature"
[image11]: ./examples/test_images.png "Test Images"
[image13]: ./examples/lane_detection.png "Lane Detection"
[video1]: ./output_videos/project_video.mp4 "Project Video"
[video2]: ./output_videos/challenge_video.mp4 "Challenge Video"
[video3]: ./output_videos/harder_challenge_video.mp4 "Harder Challenge Video"

Important Files in the project:

- **README (writeup):** Writeup with all the notes and results of the project
- **Advanced-Lane-Finding-Project.ipynb:** Notebook with all the code explaining each step of the image lane detection.
- **output_videos:** Folder with the output of the three videos.


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./Advanced-Lane-Finding-Project.ipynb".

The backbone for obtaining the distortion coefficients and calibrate the camera are the methods findChessboardCorners and calibrateCamera of OpenCV. Thea idea here is to use photos of a chess board taken from different angles and use them to find the distortion of the camera.

![image1]

First, use the findChessboardCorners on each image to find all the coordinates of the corners of the chess board. If the board contains a number of specified corners, all the points are appended to a image list so it can be used later. Once this proccess ends, is possible to use all the corner points and a predefined grid of well spaced points, to find the distortion of the image using the calibrateCamera.

The calibrateCamera function is able to return the camera distortion cooefficients that can be used to undistord an image.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The results after calibrating the camera is the next:

![image8]
![image9]

In the chess board is possible to check the difference between the original and the undistorted image easily, in the highway image you can check the position of white car to realize the difference.


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in ./Advanced-Lane-Finding-Project.ipynb) in the section "Image Process Pipeline".

The idea was to apply a set of transformation to process the image that could help to identify more easily the lane lines.

The general steps were the next:

1. Camera calibration (just one time)
2. Undistort image
3. Apply a color and gradient thresholds to generate a binary image
4. Apply perspective transformation

Once the image pass through this pipeline is ready for the lane line detection


## Color Spaces

Checking what color spaces can detect the lane lines best:

- RGB: It's possible to use the RED channel, the lane line are visible
![image4]
- HSV: It's possible to use the V channel, both lines looks good.
![image3]
- HLS: It's possible to use the S channel here, both lines looks good.
![image2]

## Sobel Gradient Absolute/Magnitude/Direction threshold

![image5]

## Creation of Binary Image 

This was one of the most important parts of the project. I tried to create a pipeline that was good in normal and light conditions. For that i try different things but at the end i did the following:

- An sobel absolute X transformation, using a very low min threshold, making this value low allows to capture more information, that is useful when is hard to detect the lane line because of ligth conditions.
- A sobel direction transformation used to reinforce the values of the lane lines using a small threshold to capture important parts of the image.
- Use the V channel of HSV to capture the lane lines, to reinforce the lane line in normal conditions (with ligther conditions it detects too much information)

The combination of the low threshold sobel absolute X transformation to detect hard ligth conditions, and use the sobel direction and V chanel to reinforce the lane line, make this pipeline works good in the three videos.


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for the perspective transform can be seen in the notebook ./Advanced-Lane-Finding-Project.ipynb in the section "Perspective Transform".

For a perspective transform is important to have two sets of points. The idea is to create a relation between thoose two sets of points to create a transformation from the "real image" to a "birde-eye view".

This was achived seleting a trapezoid shape in the "real view" (source points) and defining a rectangle with kind of the same sizes (destination poinst).

```python
src = [ [341, 650], [1078, 650], [832, 520], [511, 520]]
dst = [ [400, 650], [900, 650], [900, 500], [400, 500]]
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![image6]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The algorithm for the lane line detection goes as follows:

1. It detects the columns of the image that has more data to find the centroids where the left and right lane are.
2. Using the pre-calculated centroids, now is time to use a small window to identify the real points on each line. The window moves through the image detecting all the points on each lane.
3. Once all the points are detected is used and algorithm that can use those points to construct a mathematical function that fit all those points.
4. Using that function now is time to draw on the image the detected points on each line and draw the polynomial function that fit those points.

![image7]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

[image10]


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![image11]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Project Video ![video1] (.output_videos/project_video.mp4)

Challenge Video ![video2] (.output_videos/challenge_video.mp4)

Harder Challenge Video ![video3] (.output_videos/harder_challenge_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.
