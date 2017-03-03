
# Fovea

`fo·ve·a: A small depression in the retina of the eye where visual acuity is highest. The center of the field of vision is focused in this region, where retinal cones are particularly concentrated.`

## Introduction

Fovea provides a unified command line interface to computer vision APIs from [Google](https://cloud.google.com/vision/docs/), [Microsoft](https://www.microsoft.com/cognitive-services/en-us/computer-vision-api), [AWS](https://aws.amazon.com/rekognition/), [Clarifai](https://developer.clarifai.com/), [Imagga](https://wwww.imagga.com/), and [IBM Watson](https://www.ibm.com/watson/developercloud/visual-recognition.html). Use Fovea if you want to:

1.	Easily classify images in a shell script. See: [examples](#examples).
2.	Compare the functionality or acuity of alternative computer vision apis.

Fovea provides a standardized tabular output mode, suitable for interactive shell sessions or scripts. Where Fovea's simplified tabular output is inadequate, use its json output mode to get vendor-specific json, instead. The table below attempts to characterize the current status of Fovea's feature coverage. 

| Feature               | Google | Microsoft | Amazon | Clarifai | Watson | Imagga |OpenCV | Tabular   |  JSON |
| ---:                  |  ---   | ---       | ---    | ---      | ---    | ---    |---    |  ---      | ---  |
| [Labels](#labels) | ✅️️     | ✅    ️️   | ✅️️     |  ✅      |  ✅   | ✅       |        | ✅         ️️| ✅    ️️|
| [Label i18n](#label-i18n)    |        | ✅       |        | ✅        | ✅    | ✅       |        | ✅          | ✅      |
| [Faces](#faces)                 | ✅️️     | ✅️️       | ✅️️     |  ✅      |  ✅   |        | ✅️️     | ✅️️         | ✅️️    |
| [Landmarks](#landmarks)             | ✅     |          |        |          |        |        |        | ✅️️         | ✅️    ️|
| [Text (OCR)](#ocr)            | ✅     | ✅️️️       |        |          |        |        |        | ️️❌          | ✅️️    |
| [Emotions](#emotions)              | ✅️️     | ✅️️       | ❌️     |          |       |        |         | ❌          | ✅️️    |
| [Description](#description)           |        | ✅️️       |        |          |        |        |         | ❌          | ✅️️    |
| [Adult (NSFW)](#adult)          | ✅     | ✅️️       |        | ✅️️       |        | ❌        |         | ✅️️          | ✅️️    | 
| [Categories](#categories)            |        | ✅️️       |        |          | ✅️️     |        |         | ✅️️          | ✅️️    |
| [Image Type](#image-type)            |        | ✅️       |        |          |        |        |         | ❌          | ✅️    ️|
| [Color](#color)                 |        | ✅️️       |        | ❌       |        | ❌       |         | ❌          | ✅️️    |
| [Celebrities](#celebrities)           |        | ✅       |        | ❌       | ✅     |        |         | ❌          | ✅      |

## Installation and Setup

Clone the Fovea repository, install its dependencies, and source its environment script.

````bash
[user@host]$ git clone https://github.com/28mm/Fovea.git
[user@host]$ cd Fovea
[user@host]$ pip3 install -r requirements.txt
[user@host]$ source fovea-env.sh 
`````

Credentials are required to use web services from most providers. Most offer a rate-limited free tier for trial or low-volume users. Refer to the links below to obtain the needed credentials.

 * Google Cloud Vision API: [https://cloud.google.com/vision/docs/](https://cloud.google.com/vision/docs/)
 * Microsoft Computer Vision API: [https://www.microsoft.com/cognitive-services/en-us/computer-vision-api](https://www.microsoft.com/cognitive-services/en-us/computer-vision-api)
 * Amazon Web Services Rekognition: [https://aws.amazon.com/rekognition/](https://aws.amazon.com/rekognition/)
 * IBM Watson Image Recognition: [https://www.ibm.com/watson/developercloud/visual-recognition.html](https://www.ibm.com/watson/developercloud/visual-recognition.html)
 * Clarifai: [https://developer.clarifai.com/](https://developer.clarifai.com/)
 * Imagga: [https://docs.imagga.com](https://docs.imagga.com)

Credentials should be supplied to Fovea via environment variables. See `fovea-env.sh` for a template.

````bash
export GOOG_CV_KEY=""
export MSFT_CV_KEY=""
export AWS_CV_KEY_ID=""
export AWS_CV_KEY_SECRET=""
export AWS_CV_REGION=""
export CLARIFAI_CLIENT_ID=""
export CLARIFAI_CLIENT_SECRET=""
export CLARIFAI_ACCESS_TOKEN=""
export WATSON_CV_URL=""
export WATSON_CV_KEY=""
export IMAGGA_ID=""
export IMAGGA_SECRET=""
````

## Usage
````bash
usage: fovea [-h]
             [--provider {google,microsoft,amazon,opencv,watson,clarifai,imagga}]
             [--output {tabular,json,yaml}] [--lang LANG]
             [--ocr-lang OCR_LANG] [--labels] [--faces] [--text]
             [--emotions] [--description] [--celebrities] [--adult]
             [--categories] [--image_type] [--color] [--landmarks]
             [--confidence confidence threshold] [--model MODELS]
             [--list-models] [--list-langs] [--list-ocr-langs]
             [files [files ...]]
````

### Labels

If no other flags are set, `--labels` is set by default, and `--provider` is set to `google`. `fovea <file> [files...]`

````bash
[user@host]$ fovea http://omp.gso.uri.edu/ompweb/doee/biota/inverts/cten/pleur.jpg
0.7646665	biology
0.7288878	organism
0.616781	invertebrate
0.5076378	deep sea fish
````

The provider argument make it possible to use a different API. `fovea --provider <provider> --labels <file> [files ...]`

````bash
fovea --provider clarifai http://omp.gso.uri.edu/ompweb/doee/biota/inverts/cten/pleur.jpg
0.9975493	invertebrate
0.9917823	science
0.9688578	no person
0.968811	desktop
0.95823973	biology
[...snip...]
````

### Label i18n

Several providers offer label translations, and all default to English (en). Learn which languages a given provider supports with the list `--list-langs` flag.

````bash
[user@host]$ fovea --provider microsoft --list-langs
en
zh
````

From the list of vendor-supported languages, set the desired language with the `--lang` argument.

````bash
[user@host]$ fovea http://omp.gso.uri.edu/ompweb/doee/biota/inverts/cten/pleur.jpg --provider clarifai --lang ar
0.9975493	لافقاريات
0.9917823	العلوم
0.9688578	لا يجوز لأي شخص
0.968811	خلفية حاسوب
0.95823973	علم الاحياء
0.9574139	استكشاف
````

### Faces

Most vendors support face detection. Setting the `--face` flag will return a newline-separated list of bounding boxes. Bounding boxes are represented as four values: (left-x, top-y, width, height).

````bash
[useer@host]$ fovea --provider amazon --faces examples/face-detection/7.png
1145	51	125	125
775	112	125	125
1703	116	123	123
528	89	116	119
354	63	110	110
1506	91	106	106
55	72	101	102
````

### Landmarks

At present, only Google supports landmark and location detection. 

````bash
[user@host]$ fovea --landmarks ../ex/rattlesnake-ledge.jpg
0.35423633	Rattlesnake Lake	47.436158,-121.77812576293945
````

### OCR

OCR is only supported in the JSON output mode.

````bash
[user@host]$ fovea --text --output json turkish-road-sign.jpg
[...snip...]
                {
                    "description": "Tatlisu",
                    "boundingPoly": {
                        "vertices": [
                            {
                                "x": 174,
                                "y": 133
                            },
                            {
                                "x": 500,
                                "y": 133
                            },
                            {
                                "x": 500,
                                "y": 205
                            },
                            {
                                "x": 174,
                                "y": 205
                            }
                        ]
                    }
                },
[...snip...]
````

### Emotions

### Description

### Adult

The parameters for NSFW and Adult image detection vary a bit between vendors.

```bash
[user@host]$ fovea --adult --provider google test.jpg
0.25	nsfw

[user@host]$ fovea --adult --provider clarifai test.jpg
0.9933746	sfw
0.006625366	nsfw

[user@host]$ fovea --adult --provider microsoft test.jpg
0.12925413250923157	nsfw
0.07490675896406174	racy
````

### Categories

### Image Type

### Color

### Celebrities

## Examples

 1. [Face Detection](#face-detection)
 1. [Instagram](#instagram)
  
### Face Detection

In the example below (a still from the François Ozon film *8 femmes*), ImageMagick is used to draw a bounding box around the faces detected by six services. See `.../examples/face-detection` in the distribution directory for further details.

The Google API returns better-centered and more inclusive bounding-boxes than those from any competing services. Others are cropped more radically to include only facial features. All of the services, including the free-to-use OpenCV Haar Cascade, find all seven faces.

Google: `[user@host]$ fovea --provider google --faces file.png`
![Google](examples/face-detection/7-google.png)

Microsoft: `[user@host]$ fovea --provider microsoft --faces file.png`
![Microsoft](examples/face-detection/7-microsoft.png)



OpenCV: `[user@host]$ fovea --provider opencv --faces file.png`
![OpenCV](examples/face-detection/7-opencv.png)

Amazon: `[user@host]$ fovea --provider amazon --faces file.png`
![Rekognition](examples/face-detection/7-amazon.png)

Clarifai: `[user@host]$ fovea --provider clarifai --faces file.png`
![Clarifai](examples/face-detection/7-clarifai.png)

Watson: `[user@host]$ fovea --provider watson --faces file.png`
![Watson](examples/face-detection/7-watson.png)

### Instagram

Instagram is an interesting source of example data. Accounts are often thematic, as are hashtags. Below are the top 20 labels applied to 400 images drawn from the [kimkardashian](https://www.instagram.com/kimkardashian/?hl=en) Instagram account. 

````bash
[user@host]$ for provider in google amazon microsoft clarifai watson imagga
> do
>     fovea --labels --provider $provider *.jpg | sed 's/0\.[0-9]*[[:space:]]*//g' | sort | uniq -c | sort -n | tail -20 > labels.$provider
> done
````

| Google              | Amazon     | Microsoft | Clarifai | Watson | Imagga |
| :---              | :---         | :---         | :--- |  :---   | :--- |
|   39 nose |   43 Maillot |   11 hair |   75 brunette | 22 overgarment | 264 model |
|   39 spring |   45 Crowd |   12 water | 95 love  | 23 young lady (heroine) | 266 cute |
|   40 brown hair |   47 Accessories |   13 ground | 96 recreation | 24 entertainer | 270 youth |
|   42 face |   48 Head |   14 black |  102 group | 25 room | 267 lady | 
|   42 long hair |   59 Smile |   14 floor | 109 glamour | 25 sweetheart | 272 fashion |
|   50 lady |   60 Lingerie |   14 nature |  110 sexy | 26 alizarine red color | 268 man |
|   51 leg |   61 Bra |   15 dressed |  113 dress | 26 sister | 282 casual | 
|   55 supermodel |   62 Underwear |   16 crowd |  113 indoors |  28 maroon color | 304 women |
|   56 black hair |   66 Dress |   18 man |  116 music | 28 undergarment | 305 smiling |
|   57 model |   70 Girl |   24 beautiful |  118 facial expression | 34 Indian red color | 306 face |
|   61 photo shoot |   82 Costume |   30 sky |  124 two | 37 dress |306 happiness |
|   64 hairstyle |   86 Apparel |   30 standing |  152 man | 43 device | 320 lifestyle |
|   64 photograph |  108 Woman |   35 clothing |  163 model | 51 gray color | 321 pretty |
|   66 dress |  128 Face |   36 group |  169 one | 52 black color | 324 portrait |
|   71 hair |  139 Clothing |   52 wall |  210 girl | 60 ivory color | 331 smile |
|   76 photography |  170 Portrait |   53 people |  221 fashion | 75 female | 332 attractive |
|   96 image |  199 Selfie |   81 outdoor |  263 wear | 83 people | 333 happy |
|  103 beauty |  231 Female |   90 woman |  293 portrait | 94 woman | 336 caucasian |
|  103 clothing |  342 Human |  124 posing |  318 adult | 95 garment | 340 adult |
|  115 fashion |  344 People |  137 indoor |  337 woman | 139 coal black color |  348 person |
|  124 person |  353 Person |  302 person | 352 people | 261 person |  349 people |





