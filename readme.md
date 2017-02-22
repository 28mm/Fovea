
# Fovea

`fo·ve·a: A small depression in the retina of the eye where visual acuity is highest. The center of the field of vision is focused in this region, where retinal cones are particularly concentrated.`

## Introduction

Fovea provides a unified command line interface to computer vision APIs from Google, Microsoft, and AWS. Use Fovea if you want to:

1.	Easily classify images in a shell script. See: [examples](#Examples).
2.	Compare the functionality or acuity of alternative computer vision apis.

Fovea’s tabular output mode supports a limited subset of available features: 

1.	Classification (Labeling)
2.	Face detection
3.	Landmark/Location detection (Google only)

Where Fovea's simplified tabular output is inadequate, use json instead. Refer to the table below to see which services support which features, and which Fovea output mode supports those features.

| Feature      | Google | Microsoft | Amazon | OpenCV |   Tabular        | JSON |
| ---: |  ---      | --- | --- | --- |  ---     | --- |
| Labels | ✔️️      | ✔️️| ✔️️|       | ✔️️| ✔️️|
| Faces  |  ✔️️     | ✔️️| ✔️️| ✔️️       | ✔️️| ✔️️|
| Landmarks |  ✔️️|    |    |     | ✔️️| ✔️️|
| Text (OCR) | ✔️️| ✔️️|    |   | ️️ | ✔️️|
| Emotions | ✔️️| ✔️️|    |    |      |  ✔️️     |
| Description |   | ✔️️|    |    |       | ✔️️|
| Adult (NSFW) |        | ✔️️|  |  | | ✔️️| 
| Categories   |        | ✔️️|  |  | | ✔️️|
| Image Type   |        | ✔️ |   |  | | ✔️️|
| Color        |        | ✔️️|  |  | | ✔️️|


## Installation and Setup

Clone the Fovea repository, install its dependencies, and source its environment script.

````bash
[user@host]$ git clone https://github.com/28mm/Fovea.git
[user@host]$ cd Fovea
[user@host]$ pip3 install -r requirements.txt
[user@host]$ source fovea-env.sh 
`````


 * Google Cloud Vision API: ![https://cloud.google.com/vision/docs/](https://cloud.google.com/vision/docs/)
 * Microsoft Computer Vision API: ![https://www.microsoft.com/cognitive-services/en-us/computer-vision-api](https://www.microsoft.com/cognitive-services/en-us/computer-vision-api)
 * Amazon Web Services Rekognition: ![https://aws.amazon.com/rekognition/](https://aws.amazon.com/rekognition/)

Web services from Google, Microsoft, and Amazon require authentication. Obtain access keys for these services via the links above. Supply these keys to Fovea by setting the relevant environment variables, shown below.

````bash
export GOOG_CV_KEY=""
export MSFT_CV_KEY=""
export AWS_CV_KEY_ID=""
export AWS_CV_KEY_SECRET=""
export AWS_CV_REGION=""
````

## Usage
````bash
usage: fovea [-h] [--provider {google,microsoft,amazon,opencv}]
             [--output {tabular,json,yaml}] [--labels] [--faces] [--text]
             [--emotions] [--description] [--celebrities] [--adult]
             [--categories] [--image_type] [--color] [--landmarks]
             [--confidence confidence threshold]
             files [files ...]
````

## Examples

 1. [Face Detection](#face-detection)
 1. [Instagram](#instagram)
  
### Face Detection

Each of the API providers supported by Fovea has a face detection feature. In the example below (a still from the François Ozon film *8 femmes*), ImageMagick is used to draw a bounding box around the faces detected by each service. See `.../examples/face-detection` in the distribution directory for further details.

The Google API returns better-centered and more inclusive bounding-boxes than those from any competing services. Observe also, that all of the services, including the free-to-use OpenCV Haar Cascade, find all seven faces.

Google: `[user@host]$ fovea --provider google --faces file.png`
![Google](examples/face-detection/7-google.png)

Microsoft: `[user@host]$ fovea --provider microsoft --faces file.png`
![Microsoft](examples/face-detection/7-microsoft.png)



OpenCV: `[user@host]$ fovea --provider opencv --faces file.png`
![OpenCV](examples/face-detection/7-opencv.png)

Amazon: `[user@host]$ fovea --provider amazon --faces file.png`
![Rekognition](examples/face-detection/7-amazon.png)

### Instagram

Instagram is an interesting source of example data. Accounts are often thematic, as are hashtags. Below are the top 20 labels applied to 400 images drawn from the ![@kimkardashian](https://www.instagram.com/kimkardashian/?hl=en) Instagram account. 

````bash
[user@host]$ for provider in google microsoft amazon
> do
>     fovea --labels --provider $provider *.jpg | sed 's/0\.[0-9]*[[:space:]]*//g' | sort | uniq -c | sort -n | tail -20 > labels.$provider
> done
````

The providers mostly agree with respect to broad categories--Person is the most frequent label in all cases--but differ regarding more specific features. For example, the Google API applies numerous *hair* related labels: *hair*, *hairstyle*, *black hair* , *long hair*, and *brown hair*. The Microsoft API applies *hair*, and *hairpiece* only, and Amazon turns up ten or so *Chairs*.

| Google              | Amazon     | Microsoft |
| :---              | :---         | :---         |
|   39 nose |   43 Maillot |   11 hair | 
|   39 spring |   45 Crowd |   12 water | 
|   40 brown hair |   47 Accessories |   13 ground | 
|   42 face |   48 Head |   14 black | 
|   42 long hair |   59 Smile |   14 floor | 
|   50 lady |   60 Lingerie |   14 nature | 
|   51 leg |   61 Bra |   15 dressed | 
|   55 supermodel |   62 Underwear |   16 crowd | 
|   56 black hair |   66 Dress |   18 man | 
|   57 model |   70 Girl |   24 beautiful | 
|   61 photo shoot |   82 Costume |   30 sky | 
|   64 hairstyle |   86 Apparel |   30 standing | 
|   64 photograph |  108 Woman |   35 clothing | 
|   66 dress |  128 Face |   36 group | 
|   71 hair |  139 Clothing |   52 wall | 
|   76 photography |  170 Portrait |   53 people | 
|   96 image |  199 Selfie |   81 outdoor | 
|  103 beauty |  231 Female |   90 woman | 
|  103 clothing |  342 Human |  124 posing | 
|  115 fashion |  344 People |  137 indoor | 
|  124 person |  353 Person |  302 person |
 




