# Face Detection with Fovea


Fovea reports the bounding box for each detected face in a four field format, described below. These bounding boxes can be easily plotted over the original image, with ImageMagick, for instance.

  1. Top-X 
  2. Left-Y
  3. Width
  4. Height


Google: `[user@host]$ fovea --google --faces 7.png`
![Google](7.png-google.png)

Microsoft: `[user@host]$ fovea --microsoft --faces 7.png`
![Microsoft](7.png-microsoft.png)

OpenCV: `[user@host]$ fovea --opencv --faces 7.png`
![OpenCV](7.png-opencv.png)

Amazon: `[user@host]$ fovea --amazon --faces 7.png`
![Rekognition](7.png-amazon.png)

Clarifai: `[user@host]$ fovea --clarifai --faces 7.png`
![Clarifai](7.png-clarifai.png)

Watson: `[user@host]$ fovea --watson --faces 7.png`
![Watson](7.png-watson.png)

SightHound `[user@host]$ fovea --sighthound --faces 7.png`
![SightHound](7.png-sighthound.png)

