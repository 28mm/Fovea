#!/usr/bin/env python3

# Fovea provides a command-line interface to computer vision APIs
# from Google[1], Microsoft[2], and Amazon[3]. These services differ
# with respect to supported features and naming convention, so fovea
# implements a unified interface with simplified output. JSON output
# is supported, as well as yaml, and a simplified tabular format.
#
# TODO Facebook[4,5] support.
# TODO IBM/Watson support.
# TODO Expanded tabular output support.
# TODO Add MSFT text/ocr support
#
# [1] https://cloud.google.com/vision/docs/
# [2] https://www.microsoft.com/cognitive-services/en-us/computer-vision-api
# [3] https://aws.amazon.com/rekognition/
# [4] https://code.facebook.com/posts/457605107772545/under-the-hood-building-accessibility-tools-for-the-visually-impaired-on-facebook/
# [5] https://code.facebook.com/posts/727262460626945/under-the-hood-facebook-accessibility/

## Standard Libraries
import sys
import os
import base64
import json
import argparse
import urllib
from urllib import request
import requests
import hmac
import hashlib
from datetime import datetime
from abc import ABCMeta, abstractmethod

## 3rd Party Libraries
import yaml
import cv2
import numpy as np

# Set API Keys here, or as environment variables.
GOOG_CV_KEY        = None
MSFT_CV_KEY        = None
AWS_CV_KEY_ID      = None
AWS_CV_KEY_SECRET  = None
AWS_CV_REGION      = None
WATSON_CV_URL      = None
WATSON_CV_KEY      = None
CLARIFAI_CLIENT_ID = None
CLARIFAI_CLIENT_SECRET = None
CLARIFAI_ACCESS_TOKEN  = None
FB_USER_ID         = None
FB_PHOTOS_TOKEN    = None

############################################################################
# Utilities, Decorators, etc. ##############################################

def empty_unless(flag):
    '''Decorates instance methods. Returns an empty
    list if self.flag is False or doesnt exist'''
    def wrapper(f):
        def wrapped_fn(*args, **kwargs):
            if getattr(args[0], flag) is False:
                return []
            try:
                return f(*args, **kwargs)
            except (KeyError, IndexError):
                return []
        return wrapped_fn
    return wrapper

class BytesFile:
    '''Lazy shim, so that a Bytes can be used instead of a File
    for multipart/form-data with the Requests library.

    Maybe only need __init__(...) and read(...), since
    no evidence that other methods, attributes are used.'''

    def __init__(self, b, filename=''):
        self.b = b
        self.name = filename

    def size(self):
        return len(b)

    def read(self):
        return self.b

    def close(self):
        self.b = None

def is_png(b):
    '''Is a Bytes a PNG?'''
    return b[0] == 137 and b[1] == 80 and b[2] == 78 and b[3] == 71 \
        and b[4] == 13 and b[5] == 10 and b[6] == 26 and b[7] == 10

def is_jpg(b):
    '''Is a Bytes a JPEG?'''
    return ((b[0] << 8)|b[1]) == 65496

###########################################################################
# Main ####################################################################

def main():

    parser = argparse.ArgumentParser(description = 'Classify image contents with the the Google, Microsoft, or Amazon Computer Vision APIs.')

    # Provider Options
    parser.add_argument('--provider',
                        dest='provider',
                        choices=[ 'google', 'microsoft',  \
                                  'amazon', 'opencv',     \
                                  'watson', 'clarifai', 'facebook' ],
                        default='google')

    # Output Options
    parser.add_argument('--output',
                        dest='output',
                        choices=[ 'tabular', 'json', 'yaml' ],
                        default='tabular')

    # Query Options (after parsing, default to labels if none are set.)
    flags=[ 'labels',      # MSFT GOOG AMZN         WATSON CLARIFAI
            'faces',       # MSFT GOOG AMZN OPENCV         CLARIFAI
            'text',        # MSFT GOOG
            'emotions',    # MSFT GOOG
            'description', # MSFT
            'celebrities', # MSFT                   WATSON
            'adult',       # MSFT
            'categories',  # MSFT                   WATSON
            'image_type',  # MSFT
            'color',       # MSFT
            'landmarks' ]  #      GOOG

    for flag in flags:
        parser.add_argument('--' + flag,
                            dest=flag,
                            metavar=flag,
                            action='store_const',
                            const=True,
                            default=False)

    # Filter features based on reported confidence.
    parser.add_argument('--confidence',
                        dest='confidence',
                        metavar='confidence threshold',
                        type=float,
                        default=0.0)

    # Take one or more files (FIXME: reporting for multiple files)
    parser.add_argument('files', nargs='+')

    # Take one or more models (FIXME: Clarifai only for now)
    parser.add_argument('--model', action='append', dest='models', default=[])

    args = parser.parse_args()

    # If no `flags' are set, default to --labels.
    if True not in [ vars(args)[flag] for flag in flags ]:
        args.labels = True

    # Give precedence to credentials set as environment variables.
    for cred in ['GOOG_CV_KEY', 'MSFT_CV_KEY', 'AWS_CV_KEY_ID',
                 'AWS_CV_KEY_SECRET', 'AWS_CV_REGION',
                 'FB_USER_ID', 'FB_PHOTOS_TOKEN',
                 'WATSON_CV_URL', 'WATSON_CV_KEY',
                 'CLARIFAI_CLIENT_ID', 'CLARIFAI_CLIENT_SECRET',
                 'CLARIFAI_ACCESS_TOKEN']:
        if cred in os.environ:
            globals()[cred] = os.environ[cred]

    #
    # Loop over each filename, make cv/api request, print appropriately.
    for fname in args.files:
        with open(fname, 'rb') as f:
            query = None
            image = f.read()

            if args.provider == 'google':
                query = Google(image,
                            GOOG_CV_KEY,
                            labels=args.labels,
                            landmarks=args.landmarks,
                            faces=args.faces,
                            text=args.text)

            elif args.provider == 'microsoft':
                query = Microsoft(image,
                               MSFT_CV_KEY,
                               labels=args.labels,
                               faces=args.faces,
                               emotions=args.emotions,
                               description=args.description,
                               categories=args.categories,
                               image_type=args.image_type,
                               color=args.color,
                               adult=args.adult,
                               text=args.text,
                               celebrities=args.celebrities)

            elif args.provider == 'amazon':
                query = Amazon(image,
                               AWS_CV_KEY_ID,
                               AWS_CV_KEY_SECRET,
                               labels=args.labels,
                               faces=args.faces)

            elif args.provider == 'watson':
                query = Watson(image,
                               WATSON_CV_URL,
                               WATSON_CV_KEY,
                               labels=args.labels,
                               categories=args.categories,
                               faces=args.faces)

            elif args.provider == 'clarifai':
                query = Clarifai(image,
                                 CLARIFAI_CLIENT_ID,
                                 CLARIFAI_CLIENT_SECRET,
                                 access_token=CLARIFAI_ACCESS_TOKEN,
                                 labels=args.labels,
                                 faces=args.faces,
                                 adult=args.adult,
                                 models=args.models)

            elif args.provider == 'facebook':
                raise

            elif args.provider == 'opencv':
                query = OpenCV(image, faces=args.faces)

            query.run()

            # Select output mode and print.
            if args.output == 'tabular':
                for line in query.tabular(confidence=args.confidence):
                    print(line)
            elif args.output == 'json':
                print(query.json())
            elif args.output == 'yaml':
                print(query.yaml())

############################################################################
# Web Service Request Handling #############################################

class Query(metaclass=ABCMeta):
    """Defines a common interface for querying computer vision APIs
    from e.g. Google, Microsoft, Amazon.

    Each Query object is instantiated with an image and query parameters.
    Queries are executed via the run(...) method, and results obtained
    via the tabular(...), json(...), or yaml(...) methods."""

    @abstractmethod
    def run(self):
        raise NotImplementedError

    #
    # Pretty Printers

    @abstractmethod
    def tabular(self, confidence=0.0):
        '''Returns a list of strings to print.'''
        raise NotImplementedError

    def json(self):
        '''Returns json as a string.'''
        return json.dumps(self._json, indent=4)

    def yaml(self):
        '''Returns yaml as a string'''
        return yaml.dump(self._json)

    #
    # Possible Query Features.

    def labels(self):
        raise NotImplementedError

    def landmarks(self):
        raise NotImplementedError

    def faces(self):
        raise NotImplementedError

    def emotions(self):
        raise NotImplementedError

    def text(self):
        raise NotImplementedError

    def description(self):
        raise NotImplementedError

    def categories(self):
        raise NotImplementedError

    def image_type(self):
        raise NotImplementedError

    def adult(self):
        raise NotImplementedError

    def color(self):
        raise NotImplementedError

class Microsoft(Query):
    '''Query class for Microsoft->Cognitive Services->Cloud Vision API.'''

    _MSFT_CV_BASE_URL = 'https://api.projectoxford.ai/vision/v1.0/analyze'

    def __init__(self, image, api_key,
                 labels=True, faces=False, emotions=False, celebrities=False,
                 description=False, categories=False, image_type=False,
                 color=False, adult=False, text=False):

        self.api_key      = api_key

        self.image        = image
        self._labels      = labels
        self._faces       = faces
        self._emotions    = emotions    # Which face emoji have we got?
        self._celebrities = celebrities # Sigh.
        self._description = description # Captions
        self._categories  = categories  # Hierarchical categories.
        self._image_type  = image_type  # Clip-Art or Line Drawing?
        self._color       = color       # Accent Colors, etc.
        self._adult       = adult       # Is it Porno?
        self._text        = text        # ocr

    ########################################################################
    # Query Execution ######################################################

    def run(self):

        features = '' # Comma-Separated list of query features
        features += 'Categories,'  if self._categories  is True else ''
        features += 'Tags,'        if self._labels      is True else ''
        features += 'Description,' if self._description is True else ''
        features += 'Faces,'       if self._faces       is True else ''
        features += 'ImageType,'   if self._image_type  is True else ''
        features += 'Color,'       if self._color       is True else ''
        features += 'Adult,'       if self._adult       is True else ''
        if len(features) > 0:
            features = features.rstrip(',')

        details = '' # Details depend on chosen features
        details += 'Celebrities,' if self._celebrities is True else ''
        if len(details) > 0:
            details = details.rstrip(',')

        headers = {
            'Ocp-Apim-Subscription-Key' : self.api_key,
            'Content-Type' : 'application/octet-stream'
        }

        url = Microsoft._MSFT_CV_BASE_URL + '?visualFeatures=' + features
        if len(details) > 0:
            url += '&details=' + details
        url += '&language=en' # en is default, zh optional.

        request_obj = request.Request(url, self.image, headers)
        response    = request.urlopen(request_obj)
        response_json = json.loads(response.read().decode('UTF-8'))

        # FIXME: text/ocr is via a different endpoint.

        self._json = response_json
        return response_json

    ########################################################################
    # Munge Response data ##################################################

    @empty_unless('_labels')
    def labels(self):
        return self._json['tags']

    @empty_unless('_categories')
    def categories(self):
        return self._json['categories']

    @empty_unless('_description')
    def description(self):
        return self._json['description']['captions']

    @empty_unless('_faces')
    def faces(self):
        return self._json['faces']

    @empty_unless('_adult')
    def adult(self):
        return self._json['adult']

    ########################################################################
    # print()-able response data  ##########################################

    def tabular(self, confidence=0.0):
        r = []

        for l in self.labels():
            if l['confidence'] > confidence:
                r.append(str(l['confidence']) + '\t' + l['name'])

        for l in self.categories():
            if l['score'] > confidence:
                r.append(str(l['score']) + '\t' + l['name'])

        for l in self.description():
            if l['confidence'] > confidence:
                r.append(str(l['confidence']) + '\t' + l['text'])

        # FIXME apply confidence here, also?
        adult = self.adult() # FIXME do better
        if adult != []:
            r.append(str(adult['adultScore']) + '\t' + 'nsfw')
            r.append(str(adult['racyScore'])  + '\t' + 'racy')

        for l in self.faces():
            rect = l['faceRectangle']
            r.append(str(rect['left']) + '\t' + str(rect['top'])  \
                     + '\t' + str(rect['width'])                  \
                     + '\t' + str(rect['height'])) 
               #   + '\t' + l['gender']                             \
               #   + '\t' + str(l['age']))

        return r

class Amazon(Query):
    '''Query class for AWS Rekognition'''

    def __init__(self, image,
                 aws_key_id,
                 aws_key_secret,
                 aws_region='us-west-2',
                 labels=True, faces=False):

        self.AWS_KEY_ID     = aws_key_id
        self.AWS_KEY_SECRET = aws_key_secret
        self.AWS_REGION     = aws_region
        self.AWS_CV_URL     = 'rekognition.'     \
                              + self.AWS_REGION  \
                              + '.amazonaws.com'

        self.image     = image
        self.b64_image = base64.b64encode(image) # <-- FIXME just store image

        self._labels   = labels
        self._faces    = faces
        self._json     = None


        # Use OpenCV to obtain image dimensions.
        # Amazon reports face bounding boxes as ratios
        # to the image's overall dimensions.
        CV_LOAD_IMAGE_COLOR = 1 # FIXME
        img = cv2.imdecode(np.fromstring(image, np.uint8),
                                   CV_LOAD_IMAGE_COLOR)
        self.height, self.width, self.channels  = img.shape


    #########################################################################
    # A couple of key/singing methods borrowed from amzn docs ###############

    @staticmethod
    def sign(key, msg):
        return hmac.new(key, msg.encode('UTF-8'), hashlib.sha256).digest()

    @staticmethod
    def signature_key(key, date_stamp, region, service):
        kDate = Amazon.sign(('AWS4' + key).encode('utf-8'), date_stamp)
        kRegion = Amazon.sign(kDate, region)
        kService = Amazon.sign(kRegion, service)
        kSigning = Amazon.sign(kService, 'aws4_request')
        return kSigning

    ########################################################################
    # Query Execution ######################################################

    def run(self):

        # all of this request signing is a bit murky.
        # should probably just use boto3

        self._json = {}
        targets = { 'RekognitionService.DetectLabels' : self._labels,
                    'RekognitionService.DetectFaces'  : self._faces }

        for target, flag in targets.items():

            if flag is not True:
                continue

            now        = datetime.utcnow()
            amz_date   = now.strftime('%Y%m%dT%H%M%SZ')
            date_stamp = now.strftime('%Y%m%d') # strip time for cred scope

            method       = 'POST'
            content_type = 'application/x-amz-json-1.1'
            amz_target   = target
            host         = self.AWS_CV_URL
            endpoint     = 'https://' + host
            service      = 'rekognition'

            canonical_uri = '/'
            canonical_querystring = ''
            canonical_headers =   'content-type:' + content_type  + '\n'     \
                                  + 'host:'         + host        + '\n'     \
                                  + 'x-amz-date:'   + amz_date    + '\n'     \
                                  + 'x-amz-target:' + amz_target  + '\n'

            signed_headers = 'content-type;host;x-amz-date;x-amz-target'

            request_data = {
                # FIXME Image can't be larger than 5MB
                'Image' : { 'Bytes' : self.b64_image.decode('UTF-8') }
            }
            request_json = json.dumps(request_data)
            payload_hash = hashlib.sha256(request_json.encode('UTF-8')).hexdigest()
            canonical_request =   method + '\n'                 \
                                + canonical_uri + '\n'          \
                                + canonical_querystring + '\n'  \
                                + canonical_headers + '\n'      \
                                + signed_headers + '\n'         \
                                + payload_hash

            algorithm = 'AWS4-HMAC-SHA256'
            credential_scope =   date_stamp + '/'         \
                               + self.AWS_REGION + '/'    \
                               + service + '/'            \
                               + 'aws4_request'

            string_to_sign =    algorithm + '\n'        \
                             +  amz_date + '\n'         \
                             +  credential_scope + '\n' \
                             +  hashlib.sha256(canonical_request.encode('UTF-8')).hexdigest()

            signing_key = Amazon.signature_key(self.AWS_KEY_SECRET,
                                               date_stamp,
                                               self.AWS_REGION,
                                               service)
            signature   = hmac.new(signing_key,
                                   string_to_sign.encode('utf-8'),
                                   hashlib.sha256).hexdigest()

            authorization_header = algorithm + ' '                         \
                                   + 'Credential=' + self.AWS_KEY_ID + '/' \
                                   + credential_scope + ', '               \
                                   + 'SignedHeaders=' + signed_headers     \
                                   + ', '                                  \
                                   + 'Signature=' + str(signature)

            headers = {
                'Content-Type'  : content_type,
                'X-Amz-Date'    : amz_date,
                'X-Amz-Target'  : amz_target,
                'Authorization' : authorization_header
            }

            response = requests.post(endpoint,
                                     data=request_json,
                                     headers=headers)
            response_json = json.loads(response.text)

            self._json[target] = response_json


    ########################################################################
    # Munge Response data ##################################################

    @empty_unless('_labels')
    def labels(self):
        return self._json['RekognitionService.DetectLabels']['Labels']

    @empty_unless('_faces')
    def faces(self):
        return self._json['RekognitionService.DetectFaces']['FaceDetails']

    def tabular(self, confidence=0.0):
        '''Returns a list of strings to print.'''
        r = []
        for label in self.labels():
            if label['Confidence'] / 100. > confidence:
                r.append(str(label['Confidence'] / 100.) + '\t' + label['Name'])

        # Rekognition gives bounding box locations and dimensions
        # as ratios to overal image dimensions.
        for face in self.faces():
            box = face['BoundingBox']
            r.append(str(int(box['Left']     * self.width))   + '\t' \
                     + str(int(box['Top']    * self.height))  + '\t' \
                     + str(int(box['Width']  * self.width))   + '\t' \
                     + str(int(box['Height'] * self.height)))
        return r

class Google(Query):
    '''Query class for the Google Cloud Vision API.'''

    def __init__(self, image, api_key,
                 labels=True, landmarks=False, faces=False,
                 text=False, confidence=0.0):

        self._GOOG_CV_URL = "https://vision.googleapis.com/" \
                           +"v1/images:annotate?key="        \
                           + api_key

        self.image     = image
        self.b64_image = base64.b64encode(image) # <-- FIXME just store image
        self._labels    = labels
        self._landmarks = landmarks
        self._faces     = faces
        self._text      = text
        self._json      = None


    ########################################################################
    # Query Execution ######################################################

    def run(self):
        max_results = 10
        features    = []

        if self._labels is True:
            features.append({ 'type' : 'LABEL_DETECTION',
                              'maxResults' : max_results })
        if self._landmarks is True:
            features.append({ 'type' : 'LANDMARK_DETECTION',
                              'maxResults' : max_results })
        if self._faces is True:
            features.append({ 'type' : 'FACE_DETECTION',
                              'maxResults' : max_results })
        if self._text is True:
            features.append({ 'type' : 'TEXT_DETECTION',
                              'maxResults' : max_results })

        request_data = { 'requests' : [
            { 'image'    : { 'content' : self.b64_image.decode('UTF-8') },
              'features' : features } ]}

        request_json = json.dumps(request_data)
        request_obj = request.Request(self._GOOG_CV_URL,
                                      bytes(request_json, encoding='UTF-8'),
                                      {'Content-Type' : 'application/json'})
        response = request.urlopen(request_obj)
        response_json = json.loads(response.read().decode('UTF-8'))

        self._json = response_json
        return response_json

    ########################################################################
    # Munge Response data ##################################################

    @empty_unless('_landmarks')
    def landmarks(self):
        return self._json['responses'][0]['landmarkAnnotations']

    @empty_unless('_labels')
    def labels(self):
        return self._json['responses'][0]['labelAnnotations']

    @empty_unless('_faces')
    def faces(self):
        return self._json['responses'][0]['faceAnnotations']

    ########################################################################
    # print()-able response data  ##########################################

    def tabular(self, confidence=0.0):
        '''Returns a list of strings to print.'''
        r = []
        for l in self.labels():
            if l['score'] > confidence:
                r.append(str(l['score']) + '\t' + l['description'])

        for l in self.landmarks():
            if l['score'] > confidence:
                lmark = str(l['score']) + '\t' + l['description']

                for loc in l['locations']:
                    lmark +=  '\t' + str(loc['latLng']['latitude'])    \
                              + ',' + str(loc['latLng']['longitude'])

                r.append(lmark)

        for f in self.faces():
            vertices = f['boundingPoly']['vertices']

            min_x = sys.maxsize
            min_y = sys.maxsize
            max_x = 0
            max_y = 0

            for vertex in vertices: # Assemble a bounding rectangle.
                max_x = vertex['x'] if vertex['x'] > max_x else max_x
                max_y = vertex['y'] if vertex['y'] > max_y else max_y
                min_x = vertex['x'] if vertex['x'] < min_x else min_x
                min_y = vertex['y'] if vertex['y'] < min_y else min_y

            r.append(str(min_x) + '\t' + str(min_y) + '\t' \
                     + str(max_x - min_x) + '\t' \
                     + str(max_y - min_y) + '\t')

        return r


class OpenCV(Query):
    # http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html

    def __init__(self, image, faces=False):
        CV_LOAD_IMAGE_GRAY  = 0 # FIXME
        CV_LOAD_IMAGE_COLOR = 1 # FIXME
        self.image  = cv2.imdecode(np.fromstring(image, np.uint8),
                                   CV_LOAD_IMAGE_COLOR)
        self._faces = faces
        self._json = {}

    def run(self):

        if self._faces is True:
            cascade = cv2.CascadeClassifier(
                os.environ['FOVEADIR'] + '/opencv/haarcascade_frontalface_default.xml')

            gray  = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.3, 5)

            self._json['faces'] =                        \
                    [ {'x' : int(x), 'y' : int(y),
                       'w' : int(w), 'h' : int(h)}       \
                      for x, y, w, h in faces ]

    @empty_unless('_faces')
    def faces(self):
        return self._json['faces']

    def tabular(self, confidence=0.0):
        '''Returns a list of strings to print.'''
        r = []

        r += [ str(f['x']) + ' ' +
               str(f['y']) + ' ' +
               str(f['w']) + ' ' +
               str(f['h']) for f in self.faces() ]

        return r

class Watson(Query):

    def __init__(self, image, api_url, api_key, labels=True, categories=False, faces=False):
        self.api_url     = api_url
        self.api_key     = api_key
        self.image       = image
        self._labels     = labels
        self._categories = categories
        self._faces      = faces
        self._json       = None

    def run(self):

        self._json = {}
        # FIXME do this in __init__(...)
        if is_jpg(self.image):
            image_mime  = 'image/jpeg'
            image_fname = 'name.jpg'
        elif is_png(self.image):
            image_mime = 'image/png'
            image_fname = 'name.png'
        else:
            raise # FIXME better checking of supported filetypes.

        params = {
            'api_key' : self.api_key,
            'version' : '2016-05-20'
        }

        f = BytesFile(self.image, filename=image_fname)


        # labels, categories are via one endpoint
        if self._labels or self._categories:
            url = self.api_url + '/v3/classify'
            response = requests.post(url,
                                     params=params,
                                     files={ 'images_file' :
                                             (f.name, f, image_mime) })

            response_json = json.loads(response.text)
            self._json['labels'] = response_json

        # faces are via another endpoint
        if self._faces:
            url = self.api_url + '/v3/detect_faces'
            response = requests.post(url,
                                     params=params,
                                     files={ 'images_file' :
                                             (f.name, f, image_mime) })
            response_json = json.loads(response.text)
            self._json['faces'] = response_json



    @empty_unless('_labels')
    def labels(self):
        return [ y for x in \
                 self._json['labels']['images'][0]['classifiers'] \
                 for y in x['classes'] ]

    @empty_unless('_categories')
    def categories(self):
        return [ y for x in \
                 self._json['labels']['images'][0]['classifiers'] \
                 for y in x['classes'] if 'type_hierarchy' in y ]

    @empty_unless('_faces')
    def faces(self):
        return [ x['face_location'] for x in \
                 self._json['faces']['images'][0]['faces']  ]

    def tabular(self, confidence=0.0):
        '''Returns a list of strings to print.'''
        r = []

        for label in self.labels():
            if label['score'] > confidence:
                r.append(str(label['score']) + '\t' + label['class'])

        for label in self.categories():
            if label['score'] > confidence:
                r.append(str(label['score']) + '\t' \
                         + label['type_hierarchy'])

        for f in self.faces():
            r.append(str(f['left'])     + '\t'    \
                     + str(f['top'])    + '\t'    \
                     + str(f['width'])  + '\t'    \
                     + str(f['height']))

        return r

class Clarifai(Query):

    # dict generated with .../utilities/clarifai/models.py
    models = {
        'general-v1.3'    : 'aaa03c23b3724a16a56b629203edc62c',
        'food-items-v1.0' : 'ecf02b31ad884d6785a9bd950f8e986c',
        'apparel'         : 'e0be3b9d6a454f0493ac3a30784001ff',
        'weddings-v1.0'   : 'c386b7a870114f4a87477c0824499348',
        'travel-v1.0'     : 'eee28c313d69466f836ab83287a54ed9',
        'celeb-v1.3'      : 'e466caa0619f444ab97497640cefc4dc',
        'color'           : 'eeed0b6733a644cea07cf4c60f87ebb7',
        'nsfw-v1.0'       : 'e9576d86d2004ed1a38ba0cf39ecb4b1'
    }

    def __init__(self, image, client_id, client_secret, access_token="",
                 labels=True, faces=False, adult=False, models=[]):

        self.client_id     = client_id
        self.client_secret = client_secret
        self.access_token  = access_token
        self.image         = image
        self._labels       = labels
        self._faces        = faces
        self._adult        = adult
        self._json         = None
        self._models       = [ 'general-v1.3' ] if models is [] else models

        # Use OpenCV to obtain image dimensions.
        # Clarifai reports face bounding boxes as ratios
        # to the image's overall dimensions.
        CV_LOAD_IMAGE_COLOR = 1 # FIXME
        img = cv2.imdecode(np.fromstring(image, np.uint8),
                                   CV_LOAD_IMAGE_COLOR)
        self.height, self.width, self.channels  = img.shape


    def run(self):

        #if self._labels and 'general-v1.3' not in self._models:
        #    self._models.append('general-v1.3')

        if self._adult and 'nsfw-v1.0' not in self._models:
            self._models.append('nsfw-v1.0')
            self._labels = True # bit of a hack to allow printing.

        self._json = {}

        headers = { 'Authorization' : 'Bearer '  + self.access_token,
                    'Content-Type'  : 'application/json' }

        data = {
            'inputs' : [
                { 'data' :
                  { 'image' :
                    { 'base64' : base64.b64encode(self.image).decode('UTF-8')}
                  }
                }
            ]
        }

        for model in self._models:
            model_id = self.models[model]

            url = 'https://api.clarifai.com/v2/models/'\
                  + model_id + '/outputs' 

            response = requests.post(url,
                                     headers=headers,
                                     data=json.dumps(data))

            response_json = json.loads(response.text)

            # If we've run a second model, merge new concepts
            if not self._json:
                self._json['labels'] = response_json
            else:
                old_concepts = self._json['labels']['outputs'][0]['data']['concepts']
                new_concepts = response_json['outputs'][0]['data']['concepts']

                for c in new_concepts:
                    name  = c['name']
                    value = c['value']
                    match = False

                    for oc in old_concepts:
                        if oc['name'] == name:
                            oc['value'] = value if value > oc['value'] \
                                          else oc['value']
                            match = True
                    if match is False:
                        old_concepts.append(c)

        if self._faces:

            # FIXME find model id mappings. This is face detection?
            url = 'https://api.clarifai.com/v2/models/' \
                  + 'a403429f2ddf4b49b307e318f00e528b/outputs' 

            response = requests.post(url,
                                     headers=headers,
                                     data=json.dumps(data))

            response_json = json.loads(response.text)
            self._json['faces'] = response_json




    @empty_unless('_labels')
    def labels(self):
        return self._json['labels']['outputs'][0]['data']['concepts']

    @empty_unless('_faces')
    def faces(self):
        return self._json['faces']['outputs'][0]['data']['regions']

    def tabular(self, confidence=0.0):
        r = []

        for label in self.labels():
            if label['value'] > confidence:
                r.append(str(label['value']) + '\t' + label['name'])

        for region in self.faces():
            f = region['region_info']['bounding_box']
            left   = int(f['left_col'] * self.width)
            top    = int(f['top_row'] * self.height)
            width  = int((f['right_col'] * self.width) - left)
            height = int((f['bottom_row'] * self.height) - top)
            r.append(str(left) + '\t' + str(top) + '\t' \
                     + str(width) + '\t' + str(height))
        return r

class Facebook(Query):
    '''Stub. Should be possible to get labels from <img alt=""> as
    Well as face bounding boxes and suggested names...'''

    def __init__(self, image,
                 fb_user_id,
                 api_token,
                 fb_album='cv album',
                 labels=True):

        self._FB_PHOTOS_TOKEN = api_token
        self._FB_PHOTOS_ALBUM = album
        self._labels          = labels
        self._json            = None

        def run(self):
            raise

        def upload(self):
            raise

if __name__ == '__main__': main()
