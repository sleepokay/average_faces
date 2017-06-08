# Averaging Faces

I was interested in averaging the profile photos of participants at the [Recurse Center](https://www.recurse.com). A naive averaging of pixels yielded a blurry picture, no good. I enlisted the help of dlib, a machine learning library, and cv2, a computer vision library, to detect faces in the photos, and generate landmark points for the major features of each face. Using the facial landmarks, I generated delaunay triangulations for each profile photo, which enabled me to align the faces before averaging the pixel intensities.

Though obviously not as rigorous as straight-up statistics, an average face gives an immediate, intuitive impression of the demographic representation of a group. One of the primary goals of the Recurse Center is to foster a diverse programming community, especially for women and people of color. As the average face shows, there is probably more work to do in that regard.

The naive average:

![naive average](https://github.com/sleepokay/average_faces/blob/master/misc/testing.png)

The aligned average:

![aligned average](https://github.com/sleepokay/average_faces/blob/master/results/00-average.png)


In the results folder are also averages for each "batch" of Recursers.