## Advanced Lane Finding Project 
   
---

-- RESUBMISSION COMMENTS AT THE END --


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
[video4]: ./output_videos/resubmission_project_video.mp4 "Resubmission Video"

Important Files in the project:

- **README (writeup):** Writeup with all the notes and results of the project
- **Advanced-Lane-Finding-Project.ipynb:** Notebook with all the code for explaining each step of the image lane detection.
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

First, use the findChessboardCorners on each image to find all the coordinates of the corners of the chess board. If the board contains a number of specified corners, all the points are appended to an image list so it can be used later. Once this process ends, is possible to use all the corner points and a predefined grid of well-spaced points, to find the distortion of the image using the calibrateCamera method.

The calibrateCamera function is able to return the camera distortion coefficients that can be used to undistord an image.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The results after calibrating the camera are the next:

![image8]
![image9]

In the chess board is possible to check the difference between the original and the undistorted image easily, in the highway image you can check the position of white car to realize the difference.


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps in the notebook ./Advanced-Lane-Finding-Project.ipynb) in the section "Image Process Pipeline".

The idea was to apply a set of transformation to process the image that could help to identify more easily the lane lines.

The general steps were the next:

1. Camera calibration (just one time)
2. Undistort image
3. Apply a color and gradient thresholds to generate a binary image
4. Apply perspective transformation

Once the image pass through this pipeline is ready for the lane line detection

### Color Spaces

Checking what color spaces can detect the lane lines best:

![image4]
![image3]
![image2]

In RGB it's possible to use the RED channel, the lane lines are visible, in HSV: It's possible to use the V channel, both lines look good and in HLS: it's possible to use the S channel here, both lines looks good.

At the end I only use the V channel from HSV and the S channel from HLS.

## Sobel Gradient Absolute/Magnitude/Direction threshold

![image5]

### Creation of Binary Image 

This was one of the most important parts of the project. I tried to create a pipeline that was good in normal and light conditions. For that I try different things but at the end I did the following:

- A sobel absolute X transformation, using a very low min threshold, making this value low allows to capture more information, that is useful when is hard to detect the lane line because of light conditions. **ksize=21, threshold=(10,100)**
- A sobel direction transformation used to reinforce the values of the lane lines using a small sized threshold to capture important parts of the image. **ksize=21, threshold=(1.27,1.3)**
- Use the V channel of HSV to capture the lane lines, to reinforce the lane line in normal conditions (with lighter conditions it detects too much information) **threshold=(150,255)**

The combination of the low threshold sobel absolute X transformation to detect hard light conditions, and use the sobel direction and V channel to reinforce the lane line, make this pipeline works good in the three videos.

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for the perspective transform can be seen in the notebook ./Advanced-Lane-Finding-Project.ipynb in the section "Perspective Transform".

For a perspective transform is important to have two sets of points. The idea is to create a relation between those two sets of points to create a transformation from the "real image" to a "bird-eye view".

This was achieved selecting a trapezoid shape in the "real view" (source points) and defining a rectangle with kind of the same sizes in the “warped” image (destination points).

```python
src = [ [341, 650], [1078, 650], [832, 520], [511, 520]]
dst = [ [400, 650], [900, 650], [900, 500], [400, 500]]
```
I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![image6]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code for this is on the notebook ./Advanced-Lane-Finding-Project.ipynb in the section "Finding Lane Line".

The algorithm for the lane line detection goes as follows:

1. It detects the columns of the image that has more data to find the centroids where the left and right lane are.
2. Using the pre-calculated centroids, now is time to use a sliding window to identify the real points on each line. The window moves through the image detecting all the points on each lane.
3. Once all the points are detected and is used algorithm that can construct a mathematical function that fit the points.
4. Using that function now is time to draw on the image the detected points on each line and draw the polynomial function that fit the points.
5. Draw also a filled polygon with color green between the two lines.

![image13]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The code for this is on the notebook ./Advanced-Lane-Finding-Project.ipynb in the section "Radius of Curvature and Distance from Lane Center Calculation".

To calculate the curvature and position of the vehicle was necessary to create a relation between pixels and meters: **(transform between real world and pixel world values)**

```python
ym_per_pix = 3.048/100
xm_per_pix = 3.7/378
```
### Calculate curvature

First calculate the polynomial function of each line but this time using the transformation scalars previously defined.

```python
left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
```    
Using a formula of the radius curvature.  (using first and second derivate of a curve function)

```python
left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
```
This values are finally divided to 1000 to get the distance in km.

### Calculate position of the car respect to the center

This is done by creating again a fit function using the lane points and the scalars of the meter’s pixel transformation. **(transform between real world and pixel world coordinates)**

```python
left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
```    

The half of the image is used as real vehicle center.

```python
vehicleCenter = xMax / 2
```   

Evaluate the fit function for each line and find the middle point between them.

```python
lineLeft = left_fit_m[0]*yMax**2 + left_fit_m[1]*yMax + left_fit_m[2]
lineRight = right_fit_m[0]*yMax**2 + right_fit_m[1]*yMax + right_fit_m[2]
lineMiddle = lineLeft + (lineRight - lineLeft)/2
```   
Finally, just subtract the real value (fixed as the middle of the image) and the calculated car position.

```python
diffFromVehicle = lineMiddle - vehicleCenter
```   

![image10]

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The code for this is on the notebook ./Advanced-Lane-Finding-Project.ipynb in the section "Process Test images" and in the "Videos" section.

Using all the previous methods I create a pipeline to process each image. 

At the end the function returns a two cell image, one with the real image with the green area between lanes, and the warped image with the polynomial fit. This was a way to debug more easily which parts of the videos were more difficult for the algorithm to process.

I pass all the test images through the pipeline and the results are the next.

![image11]

All the test images were correctly processed. 
---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

**Project Video ![video1] (.output_videos/project_video.mp4):** The project video works good, the transition between frames is smooth and only has some problems and the end.

**Challenge Video ![video2] (.output_videos/challenge_video.mp4):** The challenge video has some light difficulties (the white lanes where hard to detect, and there were some shadows). The result video in general behaves almost as good as the project video.

**Harder Challenge Video ![video3] (.output_videos/harder_challenge_video.mp4):** The harder challenge video had harder light conditions and has lots of curves. The result video is not too bad, it manages to detect most of the curves and almost in all the video can detect correctly the lane lines.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The project was long with lots of components and functions, but the most critic one was the generation of the binary image using the sobel and color space transformation. I had to play with different combinations of parameters to finally get one that could behave "good" in the three videos.</br>

I probably could have tried to tune those parameters only for the project video and make it work perfectly, but I wanted to, process also the challenge and the harder challenge video. </br>

In general, the behavior of the three videos is good, maybe is necessary to use different pipeline processing functions to different light conditions to make sure that the video is able to process well in all conditions. I tried to create a general purpose pipeline but maybe is not the best approach. 
 
 # RESUBMISSION NOTES   
 
My first submission had some problems with the project video, but it can do a "decent" job detecting the lane lines in the other two videos. I did some changes to the pipeline to get better results in the project video (now is working very good) but after those changes the "challenge video" and the "harder challenge video" stop working. 

**Resubmission Video ![video4] (.output_videos/resubmission_project_video.mp4):** The lane lines are correctly detected in all the video, the changes between frames is smooth and the pipeline is able to accurately process the light changes.

The changes in the pipeline were:

- The LAB color space seems to detect the yellow color better than the other color spaces.
- The binary image is a combination of a low threshold sobel in X orientation and the color spaces, "L" from LAB "V" from HSV and "R" from "RGB".

```python
combined_binary[( ((sobel_binary == 1))  & ((l_channel_binary == 1) | (v_channel_binary == 1) | (r_channel_binary == 1)) )] = 1
```   
- I'm using a sized queue to store the last 10 results to average the output and smooth the changes between frames.

- I had to change the previous proportions used to the sum of points on the binary image (used to detect the initial points of left and right lanes) to make sure that some other objects like passing cars wouldn't deviate the calculations. 

### Questions from the review:

* What are the limitations of your particular approach? Do you think your algorithm could robustly handle nighttime driving, rain, and so on?
The model was specifically tuned for the project video, and is working poorly in the other two challenge videos that have different conditions. On rain or during night it would be the same or worst.
I do not think that a single pipeline could work for all cases, probably is necessary several of them and depending on the conditions use one, or a combinations of all.

*What are the potential failure modes? What causes them specifically and how could they be addressed?
The components used to construct the binary image are used to filter some kind of information so the lane lines can be detected. Different light conditions, obstacles in the road, the colors of the lane line or the curvature in the road, implies different parameters and pipelines for the image processing.

There are lots of possible failures points given by all the possible conditions a car can deal in the road. I think that for a algorithm to detect lane lines (using the image processing tools on this project) it would be necessary to create several types of pipelines to process different kinds of situations (light conditions, rain, low visibility etc). Some-kind of fusion between all the results of those pipelines to get the final result.

*What sort of improvements do you think are necessary before you'd feel safe riding in a car that was running your algorithm?

- Create several types of pipeline to detect different aspects and conditions of the road
- Create some function that is able to evaluate and validate those pipelines. (give some kind of score to the result of each pipeline)
- Create a "fusion" function that average all the valid solutions to create a final result.
- Test the algorithm in lots of types of conditions to make sure that behaves well under different conditions.

