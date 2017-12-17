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
[image7]: ./examples/lane_detection.png "Lane Detection"
[image8]: ./examples/undistorted_img1.png "Undistorted Image 1"
[image9]: ./examples/undistorted_img2.png "Undistorted Image 2"
[image10]: ./examples/curvature_img.png "Curvature"
[image11]: ./examples/test_images.png "Test Images"
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

The calibrate camera is able to return some camera cooefficients that can be used to undistord an image.

The results after calibrating the camera is the next:

![image8]
![image9]

In the chess board is possible to check the difference between the original and the undistorted image easily, in the highway image you can check the position of white car to realize the difference.


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 585, 460      | 320, 0        |
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.
