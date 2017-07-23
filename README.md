# **Finding Lane Lines on the Road** 


---

**Finding Lane Lines on the Road**

The goal in this project was to detect the lane lines in a series of images and then apply the same pipeline to videos.
We had to make use but not limited to 
 * Color scheme changing
 * Region masking
 * Canny edge detection
 * Hough transformation


[image1]: writeup_images/plot1.png
[image2]: writeup_images/plot2.png
[image3]: writeup_images/plot3.png
[image4]: writeup_images/plot4.png
[image5]: writeup_images/plot5.png
[image6]: writeup_images/plot6.png
[image7]: writeup_images/plot7.png
[image8]: writeup_images/plot8.png
[image9]: writeup_images/plot9.png
[image10]: writeup_images/plot10.png
[image11]: writeup_images/plot11.png
[image12]: writeup_images/plot12.png
[image13]: writeup_images/plot13.png
[channel1]: writeup_images/channels_over_line1.png
[channel2]: writeup_images/channels_over_line2.png
[channel3]: writeup_images/channels_over_line3.png
[channel4]: writeup_images/channels_over_line4.png
[channel5]: writeup_images/channels_over_line5.png

---

### Reflection

### Exploration
Let's first visualize the images, where we have to detect the lines.
![1][image1]
The images are represented on computer as a set of numbers for different channels. The very commonly used size for each
channel is 8 bits and it allows to store 255 intensity values for a channel. In RGB color model, we have 255 levels for 
each of the red, green and blue channels. A human eye does not detect very small changes in intensities, but 
higher the difference, easier it is for us to notice. We can see that the yellow and white colors create a contrast with 
the rest of the road and we would expect higher intensity values for those parts of the images. Let's check this idea.
We can plot a horizontal line on original image and plot each of the colors.
![2][channel1]
Indeed, there are peaks on the intensity plots for channels when the red line on image plot crosses the bright parts. 
Using this, we can mask the intensities that are not interesting to us and plot the resulting images
![3][image2]
Looks impressive, but not so fast! Let's see what happens when the life is not so shiny! For that, load and 
plot another set of images (screenshots from challenge.mp4 video)
![4][image3]
![5][channel2]
The road looks quite bright in two of the images and in some parts there is shadow, let's see how it affects our color masking
![6][image4]
Oops! Now we get more than we asked for! We can see it also below
![7][channel3]
If we were to decide whether we are standing on a lane line just by checking the intensities of RGB channels, 
we would have been in trouble. Now we see peaks not only at lane lines, but in many places and it seems that RGB 
channels are not very helpful in this case. Luckily, we can convert our images into another color model, where 
we have an access at least to the lightness of regions. Let's try HSL which stands for hue, saturation and 
lightness (or luminosity). Useful link to get more info on [Wikipedia](https://en.wikipedia.org/wiki/HSL_and_HSV)
![8][image5]
And what if we consider line crossing lane lines?
![9][channel4]
Now, by selecting correct intervals for channels, we can clearly distinguish lane lines from the rest of the road.
![10][image6]
Just for the sake of completeness, also for the red line crossing lane lines:
![11][channel5]
 Much better! At this point we can convert the images to grayscale, without worrying about losing anything.
![12][image7]
Let's check the "easy" images as well
![13][image8]
Now that they look similar, we can work on them together. After this point, we can apply Gaussian blur to remove the
noise and smooth the visualization 
![14][image9]
### Detecting the edges
So far, so good. But there is still an open question. How does one actually detect the line? Obviously, 
we can see it directly in the grayscale images. The intensity of gray color varies over the picture and 
our eyes are able to spot the differences in intensities. Is there a way to explain it to computer as well?
 Math has an answer to lots of questions! Difference of intensity between pixels is what we need and this is 
 nothing else but derivative of intensity value with respect to pixels. Starting somewhere in black region 
 in picture and crossing the white lines makes the gradient jump and we can consider it to spot the edges
  where these jumps happen. But instead of implementing it by ourselves, we just use the Canny edge detection 
  algorithm, which in principle, uses this logic. [Here](http://docs.opencv.org/trunk/da/d22/tutorial_py_canny.html)
   is a link to a small tutorial on opencv and [Wikipedia](https://en.wikipedia.org/wiki/Canny_edge_detector) 
   page explaining it in more detail.
![15][image10]
 As we can see, besides the line edges, we have also discovered some other ones that we don't need. 
 We can easily mask those regions
![16][image11]
### Finding lines
 
Once we have found edges, we can apply [Hough transformation](http://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/hough_lines/hough_lines.html) 
to find lines. What does it do? It transforms the points in Cartesian coordinate system into a 
Polar coordinate system to find the straight lines (not only that, but we are looking for lines here). 
Each point in Cartesian coordinate system corresponds
 to a sine curve in Polar coordinate system. When two sine curves intersect at some point, there are two 
 points in Cartesian coordinate system and a line crosses those two. Using this, we can pick up those sine 
 curve intersection points, where the number of these sinusoids are above certain threshold. Equally, there
  will be as much points on corresponding line, as sine curves that intersect. __opencv__ provides an
   implementation for utilization. Let's see if we are able to find the lines using it
![17][image12]

We can clearly detect the lines. There are several lines on both sides and it would be nice to average and extrapolate 
the lines to the ends of the region of interest. To do that (actually not only that, but rather the whole process),
 I have create two classes for convenience. The first class defined is `MeanWithQueue` and the sole 
 aim of this class is to keep a pair of values in two separate queues as a history (over last k frames).  
I use it in class `Pipeline` for smoother changes in lines. It is applied to the slope of the found lines, as well as to the 
pivoting points (or in other words, intercept) that will be used for extrapolation. At each frame, I calculate 
the current slope and pivoting points for left and right lanes using medians (statistically less prone to outliers). 
Then, the mean value both for slope and pivoting point is calculated and is used for plotting. The same is done 
separately for left and right lines. To split lines into left and right, we can just calculate the slopes and partition 
them based on their signs. To extrapolate the lines to the ends of selected region, I use the function below

```python
def _extrapolate(x1, y1, slope, x0, x2):
    """
    Extrapolate from pivot point to the ends
    :param x1:
    :param y1:
    :param slope:
    :param x0:
    :param x2:
    :return: tuple containing low and high end points
    """
    y0 = int(slope * (x0 - x1) + y1)
    y2 = int(slope * (x2 - x1) + y1)
    return (x0, y0), (x2, y2)
```
where __x1__ and __y1__ are coordinates of the pivoting point and x0 and x2 are the x coordinates of lower and 
higher ends of selected region, correspondingly.  

### Shortcomings
In general, the implementation solves the problem. Namely, it detects and plots the lines through the whole video
for all three files correctly. However, one can try to fit a polynomial after getting the ends of the lines from
Hough transformation. That way, we could plot curves instead of lines. On the other point, for these videos (actually, 
curvature occurs only in challenge.mp4 file) radius of turns is quite big and therefore, lines might suffice. 

Defining the ends of region for finding lines and channel intensity selection was done manually by trial and error. 
One could also come up with a more complicated way to automate it as well.
  
No thresholding was used for the slopes, which reduces the effort for the manual pick up of one more parameter. 
The only check is done for vertical line slopes.
 
![18][image13]

- [White Line Video on Youtube](https://www.youtube.com/watch?v=SsN22pfCdqE)
- [Yellow Line Video on Youtube](https://www.youtube.com/watch?v=6CwnLnwFQLE)
- [Challenge Video on Youtube](https://www.youtube.com/watch?v=Kcqx5Y0KvIg)
 
***

