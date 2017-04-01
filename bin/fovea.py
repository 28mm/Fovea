#!/usr/bin/env python3

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
IMAGGA_ID          = None
IMAGGA_SECRET      = None
FB_USER_ID         = None
FB_PHOTOS_TOKEN    = None
SIGHTHOUND_TOKEN   = None
FACE_PP_KEY        = None
FACE_PP_SECRET     = None

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

    # Context Manager bits (for 'with/as' blocks)
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False # "return True" suppresses Exception


def flex_open(fname):
    'open and return a file, or present a file-like proxy if given a url'
    return BytesFile(requests.get(fname).content) if is_url(fname) else \
        open(fname, 'rb')

def is_png(b):
    '''Is a Bytes a PNG?'''
    return b[0] == 137 and b[1] == 80 and b[2] == 78 and b[3] == 71 \
        and b[4] == 13 and b[5] == 10 and b[6] == 26 and b[7] == 10

def is_jpg(b):
    '''Is a Bytes a JPEG?'''
    return ((b[0] << 8)|b[1]) == 65496

def is_url(b):
    '''Is a Bytes an HTTP url?'''
    return str(b[0:4]) == 'http'

def image_mime(b):
    if is_jpg(b):
        return 'image/jpeg'
    elif is_png(b):
        return  'image/png'
    else:
        raise

def image_fname(b):
    if is_jpg(b):
        return 'image.jpg'
    elif is_png(b):
        return  'image.png'
    else:
        raise
    
###########################################################################
# Main ####################################################################

def main():

    parser = argparse.ArgumentParser(description = 'Classify image contents with the the Google, Microsoft, or Amazon Computer Vision APIs.')

    dispatch_tbl = { 'google'     : Google,
                     'microsoft'  : Microsoft,
                     'amazon'     : Amazon,
                     'opencv'     : OpenCV,
                     'watson'     : Watson,
                     'clarifai'   : Clarifai,
                     'facebook'   : Facebook,
                     'imagga'     : Imagga,
                     'sighthound' : SightHound,
                     'face++'     : FacePlusPlus }

    provider_opts = [ k for k in dispatch_tbl.keys() ]
    output_opts   = [ 'tabular', 'json', 'yaml' ]

    # Provider Options
    parser.add_argument('--provider',
                        dest='provider',
                        choices=provider_opts,
                        default='google')

    # We want be able to set providers with a single flag, rather than
    # --provider <provider>...
    pro_group = parser.add_mutually_exclusive_group()
    for k in dispatch_tbl.keys():
        pro_group.add_argument('--'+ k,
                               dest=k,
                               metavar=k,
                               action='store_const',
                               const=True,
                               default=False)

    # Output Options
    parser.add_argument('--output',
                        dest='output',
                        choices=output_opts,
                        default='tabular')

    # We want be able to set output mode with a single flag, rather than
    # --output <provider>...
    out_group = parser.add_mutually_exclusive_group()
    for opt in output_opts:
        out_group.add_argument('--'+ opt,
                              dest=opt,
                              metavar=opt,
                              action='store_const',
                              const=True,
                              default=False)

    # Classifier/Labels Language Options
    parser.add_argument('--lang',
                        dest='lang',
                        default='en')

    # OCR Language Options
    parser.add_argument('--ocr-lang',
                        dest='ocr_lang',
                        default=None)

    # Max Labels
    parser.add_argument('--max-labels',
                        dest='max_labels',
                        type=int,
                        default=10)

    # Precision of Confidence Scores
    parser.add_argument('--precision',
                        dest='precision',
                        type=int,
                        default=2)

    # Query Options (after parsing, default to labels if none are set.)
    flags=[ 'labels',
            'faces',
            'text',
            'emotions',
            'description',
            'celebrities',
            'adult',
            'categories',
            'image_type',
            'color',       
            'landmarks',
            'vehicles' ]

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

    # If ontology IDs are available, should we print them?
    parser.add_argument('--ontology',
                        dest='ontology',
                        action='store_const',
                        const=True,
                        default=False)

    # Take one or more files (FIXME: reporting for multiple files)
    parser.add_argument('files', nargs='*')

    # Take one or more models (FIXME: Clarifai only for now)
    parser.add_argument('--model', action='append', dest='models', default=[])

    parser.add_argument('--list-models', dest='list_models',
                        action='store_const', const=True, default=False)

    parser.add_argument('--list-langs', dest='list_langs',
                        action='store_const', const=True, default=False)

    parser.add_argument('--list-ocr-langs', dest='list_ocr_langs',
                        action='store_const', const=True, default=False)

    args = parser.parse_args()

    # Was provider set with --provider <provider> or --<provider>?
    args_dict = vars(args)
    for k in dispatch_tbl:
        if args_dict[k] is True:
            args_dict['provider'] = k

    # Was provider set with --output <output mode> or --<output mode>
    for k in output_opts:
        if args_dict[k] is True:
            args_dict['output'] = k

    if args.list_models:
        for k, v in dispatch_tbl[args.provider].models.items():
            print(k + '\t' + v)

    if args.list_langs:
        for lang in dispatch_tbl[args.provider].label_langs:
            print(lang)

    if args.list_ocr_langs:
        for lang in dispatch_tbl[args.provider].ocr_langs:
            print(lang)

    # Check for --lang support
    if args.lang != 'en':
        if args.lang not in dispatch_tbl[args.provider].label_langs:
            raise

    # Check for --ocr-lang support
    if args.ocr_lang:
        if args.ocr_lang not in dispatch_tbl[args.provider].ocr_langs:
            raise

    # If no `flags' are set, default to --labels.
    if True not in [ vars(args)[flag] for flag in flags ]:
        args.labels = True

    # Give precedence to credentials set as environment variables.
    for cred in ['GOOG_CV_KEY',
                 'MSFT_CV_KEY',
                 'AWS_CV_KEY_ID', 'AWS_CV_KEY_SECRET', 'AWS_CV_REGION',
                 'FB_USER_ID', 'FB_PHOTOS_TOKEN',
                 'WATSON_CV_URL', 'WATSON_CV_KEY',
                 'CLARIFAI_CLIENT_ID', 'CLARIFAI_CLIENT_SECRET',
                 'CLARIFAI_ACCESS_TOKEN',
                 'IMAGGA_ID', 'IMAGGA_SECRET',
                 'SIGHTHOUND_TOKEN',
                 'FACE_PP_KEY', 'FACE_PP_SECRET' ]:
        if cred in os.environ:
            globals()[cred] = os.environ[cred]

    #
    # Loop over each filename, make cv/api request, print appropriately.
    for fname in args.files:
        with flex_open(fname) as f:
            query = None
            image = f.read()

            if args.provider == 'google':
                query = Google(image,
                               GOOG_CV_KEY,
                               labels=args.labels,
                               landmarks=args.landmarks,
                               faces=args.faces,
                               text=args.text,
                               adult=args.adult,
                               max_labels=args.max_labels,
                               precision=args.precision)

            elif args.provider == 'microsoft':

                label_lang = args.lang
                ocr_lang   = args.ocr_lang if args.ocr_lang else 'unk'

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
                                  celebrities=args.celebrities,
                                  label_lang=label_lang,
                                  ocr_lang=ocr_lang,
                                  max_labels=args.max_labels,
                                  precision=args.precision)

            elif args.provider == 'amazon':
                query = Amazon(image,
                               AWS_CV_KEY_ID,
                               AWS_CV_KEY_SECRET,
                               labels=args.labels,
                               faces=args.faces,
                               max_labels=args.max_labels,
                               precision=args.precision)

            elif args.provider == 'watson':
                if args.lang not in Watson.label_langs:
                    raise

                query = Watson(image,
                               WATSON_CV_URL,
                               WATSON_CV_KEY,
                               labels=args.labels,
                               categories=args.categories,
                               faces=args.faces,
                               label_lang=args.lang,
                               max_labels=args.max_labels,
                               precision=args.precision)

            elif args.provider == 'clarifai':
                query = Clarifai(image,
                                 CLARIFAI_CLIENT_ID,
                                 CLARIFAI_CLIENT_SECRET,
                                 access_token=CLARIFAI_ACCESS_TOKEN,
                                 labels=args.labels,
                                 faces=args.faces,
                                 adult=args.adult,
                                 celebrities=args.celebrities,
                                 models=args.models,
                                 label_lang=args.lang,
                                 max_labels=args.max_labels,
                                 precision=args.precision,
                                 color=args.color)

            elif args.provider == 'facebook':
                raise

            elif args.provider == 'opencv':
                query = OpenCV(image, faces=args.faces)

            elif args.provider == 'imagga':
                query = Imagga(image,
                               { 'IMAGGA_ID' : IMAGGA_ID,
                                 'IMAGGA_SECRET' : IMAGGA_SECRET },
                               labels=args.labels,
                               ontology=args.ontology,
                               label_lang=args.lang,
                               max_labels=args.max_labels,
                               precision=args.precision)

            elif args.provider == 'sighthound':
                query = SightHound(image,
                                   SIGHTHOUND_TOKEN,
                                   faces=args.faces,
                                   emotions=args.emotions,
                                   celebrities=args.celebrities,
                                   vehicles=args.vehicles)

            elif args.provider == 'face++':
                query = FacePlusPlus(image,
                                     FACE_PP_KEY,
                                     FACE_PP_SECRET,
                                     faces=args.faces,
                                     emotions=args.emotions)

            query.run()

            # Select output mode and print.
            if args.output == 'tabular':
                for line in query.tabular(
                        confidence=args.confidence,
                        ontology=args.ontology):
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
    def tabular(self, confidence=0.0, ontology=False):
        '''Returns a list of strings to print.'''
        raise NotImplementedError

    def json(self):
        '''Returns json as a string.'''
        return json.dumps(self._json, indent=4)

    def yaml(self):
        '''Returns yaml as a string'''
        return yaml.dump(self._json)

    #
    # Class Attributes

    models = {}            # multiple model support. { canonical_name : id}
    url_support = False    # image can be given as a url
    label_langs = [ 'en' ] # supported classifier languages
    ocr_langs   = []       # supported ocr languages
    ontology_placeholder = 'xxxxxx'

    precision_fmts = { 1 : '{0:.1f}',
                       2 : '{0:.2f}',
                       3 : '{0:.3f}',
                       4 : '{0:.4f}',
                       5 : '{0:.5f}',
                       6 : '{0:.6f}',
                       7 : '{0:.7f}',
                       8 : '{0:.8f}',
                       9 : '{0:.9f}',
                       10 : '{0:.10f}' }

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

    def vehicles(self):
        raise NotImplementedError

class Microsoft(Query):
    '''Query class for Microsoft->Cognitive Services->Cloud Vision API.'''

    _MSFT_CV_BASE_URL  = 'https://api.projectoxford.ai/vision/v1.0/analyze'
    _MSFT_OCR_BASE_URL = 'https://westus.api.cognitive.microsoft.com/vision/v1.0/ocr'

    label_langs = [ 'en', 'zh' ] # supported classifier/tag languages

    # supported OCR languages
    ocr_langs   = [ 'unk',     # (AutoDetect)
                    'zh-Hans', # (ChineseSimplified)
                    'zh-Hant', # (ChineseTraditional)
                    'cs',      # (Czech)
                    'da',      # (Danish)
                    'nl',      # (Dutch)
                    'en',      # (English)
                    'fi',      # (Finnish)
                    'fr',      # (French)
                    'de',      # (German)
                    'el',      # (Greek)
                    'hu',      # (Hungarian)
                    'it',      # (Italian)
                    'ja',      # (Japanese)
                    'ko',      # (Korean)
                    'nb',      # (Norwegian)
                    'pl',      # (Polish)
                    'pt',      # (Portuguese,
                    'ru',      # (Russian)
                    'es',      # (Spanish)
                    'sv',      # (Swedish)
                    'tr'       # (Turkish)
    ]

    def __init__(self, image, api_key,
                 labels=True, faces=False, emotions=False, celebrities=False,
                 description=False, categories=False, image_type=False,
                 color=False, adult=False, text=False,
                 label_lang='en', ocr_lang='unk', max_labels=10,
                 precision=2):

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
        self.label_lang   = label_lang
        self.ocr_lang     = ocr_lang
        self.max_labels   = max_labels
        self.precision    = precision
        self.prec_fmt = Query.precision_fmts[precision]

    ########################################################################
    # Query Execution ######################################################

    def run(self):

        self._json = {}

        features = '' # Comma-Separated list of query features
        features += 'Categories,'  if self._categories  is True else ''
        features += 'Tags,'        if self._labels      is True else ''
        features += 'Description,' if self._description is True else ''
        features += 'ImageType,'   if self._image_type  is True else ''
        features += 'Color,'       if self._color       is True else ''
        features += 'Adult,'       if self._adult       is True else ''
        features += 'Faces,' if self._faces is True or self._celebrities is True else ''

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

        if len(features) > 0:
            url = Microsoft._MSFT_CV_BASE_URL + '?visualFeatures=' + features
            if len(details) > 0:
                url += '&details=' + details
            url += '&language=' + self.label_lang

            request_obj = request.Request(url, self.image, headers)
            response    = request.urlopen(request_obj)
            response_json = json.loads(response.read().decode('UTF-8'))
            self._json = response_json

        if self._text:
            url = Microsoft._MSFT_OCR_BASE_URL     \
                  + '?language=' + self.ocr_lang   \
                  + '&detectOrientation=True'
            request_obj = request.Request(url, self.image, headers)
            response    = request.urlopen(request_obj)
            response_json = json.loads(response.read().decode('UTF-8'))
            self._json['text'] = response_json

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

    @empty_unless('_celebrities')
    def celebrities(self):
        return [ x['detail']['celebrities'] for x in self._json['categories'] if x['name'].startswith('people_') ][0]

    @empty_unless('_adult')
    def adult(self):
        return self._json['adult']

    ########################################################################
    # print()-able response data  ##########################################

    def tabular(self, confidence=0.0, ontology=False):
        r = []
        l_count = 0

        for l in self.labels():
            if l['confidence'] > confidence:
                if l_count == self.max_labels:
                    break
                l_count += 1
                r.append(self.prec_fmt.format(l['confidence']) + '\t' + l['name'])

        for l in self.categories():
            if l['score'] > confidence:
                r.append(self.prec_fmt.format(l['score']) + '\t' + l['name'])

        for l in self.description():
            if l['confidence'] > confidence:
                r.append(self.prec_fmt.format(l['confidence']) + '\t' + l['text'])

        # FIXME apply confidence here, also?
        adult = self.adult() # FIXME do better
        if adult != []:
            r.append(self.prec_fmt.format(adult['adultScore']) + '\t' + 'nsfw')
            r.append(self.prec_fmt.format(adult['racyScore'])  + '\t' + 'racy')

        for l in self.faces():
            rect = l['faceRectangle']
            r.append(str(rect['left']) + '\t' + str(rect['top'])  \
                     + '\t' + str(rect['width'])                  \
                     + '\t' + str(rect['height'])) 
               #   + '\t' + l['gender']                             \
               #   + '\t' + str(l['age']))

        for l in self.celebrities():
            if l['confidence'] < confidence:
              continue

            rect = l['faceRectangle']
            o_placeholder = self.ontology_placeholder + '\t' if ontology else ''
            r.append(str(rect['left']) + '\t' + str(rect['top'])    \
                     + '\t' + str(rect['width'])                    \
                     + '\t' + str(rect['height'])                   \
                     + '\t' + self.prec_fmt.format(l['confidence']) \
                     + '\t' + o_placeholder                         \
                     + l['name'])

        return r

class Amazon(Query):
    '''Query class for AWS Rekognition'''

    def __init__(self, image,
                 aws_key_id,
                 aws_key_secret,
                 aws_region='us-west-2',
                 labels=True, faces=False,
                 max_labels=10,
                 precision=2):

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

        self.max_labels = max_labels
        self.precision  = precision
        self.prec_fmt = Query.precision_fmts[precision]

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

    def tabular(self, confidence=0.0, ontology=False):
        '''Returns a list of strings to print.'''
        r = []
        l_count = 0
        for label in self.labels():
            if l_count == self.max_labels:
                break
            if label['Confidence'] / 100. > confidence:
                l_count+=1
                r.append(self.prec_fmt.format(label['Confidence'] / 100.) \
                         + '\t' + label['Name'])

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

    # (Beware: these are arbitrary assignments!)
    likelihood = {
        'VERY_LIKELY'   : 0.99,
        'LIKELY'        : 0.75,
        'UNLIKELY'      : 0.25,
        'VERY_UNLIKELY' : 0.01
    }

    label_langs = [ 'en', 'zh' ] # supported classifier/tag languages

    # supported OCR languages
    ocr_langs = [
        'af',  # (Afrikaans)
        'ar',  # (Arabic)
        'as',  # (Assamese)
        'az',  # (Azerbaijani)
        'be',  # (Belarusian)
        'bn',  # (Bengali)
        'bg',  # (Bulgarian)
        'ca',  # (Catalan)
        'zh',  # (Chinese)
        'zh-CN',  # (Chinese)
        'zh-TW',  # (Chinese)
        'hr',  # (Croatian)
        'cs',  # (Czech)
        'da',  # (Danish)
        'nl',  # (Dutch)
        'et',  # (Estonian)
        'fi',  # (Finnish)
        'fr',  # (French)
        'de',  # (German)
        'el',  # (Greek)
        'he',  # (Hebrew)
        'hi',  # (Hindi)
        'hu',  # (Hungarian)
        'is',  # (Icelandic)
        'id',  # (Indonesian)
        'it',  # (Italian)
        'ja',  # (Japanese)
        'kk',  # (Kazakh)
        'ko',  # (Korean)
        'ky',  # (Kyrgyz)
        'lv',  # (Latvian)
        'lt',  # (Lithuanian)
        'mk',  # (Macedonian)
        'mr',  # (Marathi)
        'mn',  # (Mongolian)
        'ne',  # (Nepali)
        'no',  # (Norwegian)
        'ps',  # (Pashtu)
        'fa',  # (Persian)
        'pl',  # (Polish)
        'pt',  # (Portuguese)
        'ro',  # (Romanian)
        'ru',  # (Russian)
        'sa',  # (Sanskrit)
        'sr',  # (Serbian)
        'sk',  # (Slovak)
        'sl',  # (Slovenian)
        'es',  # (Spanish)
        'sv',  # (Swedish)
        'tl',  # (Tagalog)
        'ta',  # (Tamil)
        'th',  # (Thai)
        'tr',  # (Turkish)
        'uk',  # (Ukrainian)
        'ur',  # (Urdu)
        'uz',  # (Uzbek)
        'vi',  # (Vietnamese)
    ]

    def __init__(self, image, api_key,
                 labels=True, landmarks=False, faces=False,
                 text=False, adult=False, confidence=0.0,
                 max_labels=10, precision=2):

        self._GOOG_CV_URL = "https://vision.googleapis.com/" \
                           +"v1/images:annotate?key="        \
                           + api_key

        self.image     = image
        self.b64_image = base64.b64encode(image) # <-- FIXME just store image
        self._labels    = labels
        self._landmarks = landmarks
        self._faces     = faces
        self._text      = text
        self._adult     = adult
        self._json      = None
        self.max_labels = max_labels
        self.precision  = precision
        self.prec_fmt   = Query.precision_fmts[precision]


    ########################################################################
    # Query Execution ######################################################

    def run(self):
        features    = []
        max_results = 100

        if self._labels is True:
            features.append({ 'type' : 'LABEL_DETECTION',
                              'maxResults' : self.max_labels })
        if self._landmarks is True:
            features.append({ 'type' : 'LANDMARK_DETECTION',
                              'maxResults' : max_results })
        if self._faces is True:
            features.append({ 'type' : 'FACE_DETECTION',
                              'maxResults' : max_results })
        if self._text is True:
            features.append({ 'type' : 'TEXT_DETECTION',
                              'maxResults' : max_results })
        if self._adult is True:
            features.append({ 'type' : 'SAFE_SEARCH_DETECTION',
                              'maxResults': max_results })

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

    @empty_unless('_adult')
    def adult(self):
        return self._json['responses'][0]['safeSearchAnnotation']

    @empty_unless('_faces')
    def faces(self):
        return self._json['responses'][0]['faceAnnotations']

    ########################################################################
    # print()-able response data  ##########################################

    def tabular(self, confidence=0.0, ontology=False):
        '''Returns a list of strings to print.'''
        r = []
        for l in self.labels():
            if l['score'] > confidence:
                if ontology is True:
                    r.append(self.prec_fmt.format(l['score']) + '\t' \
                              + l['mid'] + '\t'                      \
                              + l['description'])
                else:
                    r.append(self.prec_fmt.format(l['score']) + '\t' + l['description'])

        adult = self.adult()
        if adult:
            r.append(str(max(Google.likelihood[adult['adult']],
                             Google.likelihood[adult['violence']])) + '\t' + 'nsfw')

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

    def tabular(self, confidence=0.0, ontology=False):
        '''Returns a list of strings to print.'''
        r = []

        r += [ str(f['x']) + ' ' +
               str(f['y']) + ' ' +
               str(f['w']) + ' ' +
               str(f['h']) for f in self.faces() ]

        return r

class Watson(Query):

    # supported classifier/tag languages
    label_langs = [ 'en',  # (English)
                    'es',  # (Spanish)
                    'ar',  # (Arabic)
                    'ja' ] # (Japanese)

    def __init__(self, image, api_url, api_key, labels=True, categories=False, faces=False, label_lang='en', max_labels=10, precision=2):
        self.api_url     = api_url
        self.api_key     = api_key
        self.image       = image
        self._labels     = labels
        self._categories = categories
        self._faces      = faces
        self._json       = None
        self.label_lang  = label_lang
        self.max_labels  = max_labels
        self.precision   = precision
        self.prec_fmt    = Query.precision_fmts[precision]

    def run(self):
        self._json = {}

        mime  = image_mime(self.image)
        fname = image_fname(self.image) 

        params = {
            'api_key' : self.api_key,
            'version' : '2016-05-20'
        }

        headers = {
            'Accept-Language' : self.label_lang
        }

        f = BytesFile(self.image, filename=fname)


        # labels, categories are via one endpoint
        if self._labels or self._categories:
            url = self.api_url + '/v3/classify'
            response = requests.post(url,
                                     params=params,
                                     headers=headers,
                                     files={ 'images_file' :
                                             (f.name, f, mime) })

            response_json = json.loads(response.text)
            self._json['labels'] = response_json

        # faces are via another endpoint
        if self._faces:
            url = self.api_url + '/v3/detect_faces'
            response = requests.post(url,
                                     params=params,
                                     files={ 'images_file' :
                                             (f.name, f, mime) })
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

    def tabular(self, confidence=0.0, ontology=False):
        '''Returns a list of strings to print.'''
        r = []

        l_count=0
        for label in self.labels():
            if l_count == self.max_labels:
                break
            if label['score'] > confidence:
                l_count+=1
                r.append(self.prec_fmt.format(label['score']) \
                         + '\t' + label['class'])

        for label in self.categories():
            if label['score'] > confidence:
                r.append(self.prec_fmt.format(label['score']) \
                         + '\t' \
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

    # Only supported with 'general' model. Otherwise: en, only.
    label_langs = [
        'ar',  # (Arabic)
        'bn',  # (Bengali)
        'da',  # (Danish)
        'de',  # (German)
        'en',  # (English)
        'es',  # (Spanish)
        'fi',  # (Finnish)
        'fr',  # (French)
        'hi',  # (Hindi)
        'hu',  # (Hungarian)
        'it',  # (Italian)
        'ja',  # (Japanese)
        'ko',  # (Korean)
        'nl',  # (Dutch)
        'no',  # (Norwegian)
        'pa',  # (Punjabi)
        'pl',  # (Polish)
        'pt',  # (Portuguese)
        'ru',  # (Russian)
        'sv',  # (Swedish)
        'tr',  # (Turkish)
        'zh',  # (Chinese)
        'zh-TW'  # (Chinese)
    ]

    def __init__(self, image, client_id, client_secret, access_token="",
                 labels=True, faces=False, adult=False, 
                 celebrities=False, models=[], color=False,
                 label_lang='en', max_labels=10, precision=2):

        self.client_id     = client_id
        self.client_secret = client_secret
        self.access_token  = access_token
        self.image         = image
        self._labels       = labels
        self._faces        = faces
        self._adult        = adult
        self._celebrities  = celebrities
        self._color        = color
        self._json         = None
        self._models       = [ 'general-v1.3' ] if models       == []      \
                                                and celebrities == False   \
                                                and faces       == False   \
                              else models
        self.label_lang    = label_lang
        self.max_labels    = max_labels
        self.precision     = precision
        self.prec_fmt      = Query.precision_fmts[precision]
        
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

        # FIXME multi-lang support only avaialable for
        # the general model, right now.
        if self.label_lang != 'en':
            data['model'] = { 'output_info' :
                              { 'output_config' :
                                { 'concepts_mutually_exclusive' : False,
                                  'closed_environment' : False,
                                  'language' : self.label_lang
                                }
                              }
            }

        for model in self._models:
            model_id = Clarifai.models[model]

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
                print(response_json)
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

        if self._celebrities:
          url = 'https://api.clarifai.com/v2/models/' \
               + self.models['celeb-v1.3'] + '/outputs'

          response = requests.post(url,
                                   headers=headers,
                                   data=json.dumps(data))
          response_json = json.loads(response.text)
          self._json['celebrities'] = response_json

        if self._color:
          url = 'https://api.clarifai.com/v2/models/' \
                + self.models['color'] + '/outputs'

          response = requests.post(url,
                                   headers=headers,
                                   data=json.dumps(data))
          response_json = json.loads(response.text)
          self._json['color'] = response_json


    @empty_unless('_labels')
    def labels(self):
        return self._json['labels']['outputs'][0]['data']['concepts']

    @empty_unless('_faces')
    def faces(self):
        return self._json['faces']['outputs'][0]['data']['regions']

    @empty_unless('_celebrities')
    def celebrities(self):
        return self._json['celebrities']['outputs'][0]['data']['regions']

    @empty_unless('_color')
    def color(self):
        return self._json['color']['outputs'][0]['data']['colors']

    def tabular(self, confidence=0.0, ontology=False):
        r = []

        l_count = 0
        for label in self.labels():
            if l_count == self.max_labels:
                break
            if label['value'] > confidence:
                l_count += 1
                r.append(self.prec_fmt.format(label['value']) \
                         + '\t'                          \
                         + label['name'])

        for region in self.faces():
            f = region['region_info']['bounding_box']
            left   = int(f['left_col'] * self.width)
            top    = int(f['top_row'] * self.height)
            width  = int((f['right_col'] * self.width) - left)
            height = int((f['bottom_row'] * self.height) - top)
            r.append(str(left) + '\t' + str(top) + '\t' \
                     + str(width) + '\t' + str(height))

        for region in self.celebrities():
            f = region['region_info']['bounding_box']
            left   = int(f['left_col'] * self.width)
            top    = int(f['top_row'] * self.height)
            width  = int((f['right_col'] * self.width) - left)
            height = int((f['bottom_row'] * self.height) - top)
            identities = region['data']['face']['identity']['concepts']

            for i in identities:
              if i['value'] < confidence:
                continue

              o_placeholder = i['id'] + '\t' if ontology else ''
              r.append(str(left) + '\t' + str(top) + '\t'               \
                        + str(width) + '\t' + str(height) + '\t'        \
                        + str(self.prec_fmt.format(i['value'])) + '\t'  \
                        + o_placeholder + i['name'])

        for color in self.color():
            r.append(self.prec_fmt.format(color['value']) + '\t' \
                      + color['raw_hex']    + '\t'               \
                      + color['w3c']['hex'] + '\t'               \
                      + color['w3c']['name'] )
        return r

class Imagga(Query):

    label_langs = [
        'ar', # Arabic
        'bg', # Bulgarian
        'bs', # Bosnian
        'en', # (default) 	English
        'ca', # Catalan
        'cs', # Czech
        'cy', #	Welsh
        'da', #	Danish
        'de', #	German
        'el', #	Greek
        'es', #	Spanish
        'et', #	Estonian
        'fa', #	Persian
        'fi', # Finnish
        'fr', # French
        'he', # Hebrew
        'hi', # Hindi
        'hr', # Croatian
        'ht', # Haitian Creole
        'hu', # Hungarian
        'id', # Indonesian
        'it', # Italian
        'ja', # Japanese
        'ko', # Korean
        'lt', # Lithuanian
        'lv', # Latvian
        'ms', # Malay
        'mt', # Maltese
        'mww', # Hmong Daw
        'nl', # Dutch
        'no', # Norwegian
        'otq', # Querétaro Otomi
        'pl', # Polish
        'pt', # Portuguese
        'ro', # Romanian
        'ru', # Russian
        'sk', # Slovak
        'sv', # Swedish
        'sl', # Slovenian
        'sr_cyrl', # Serbian - Cyrillic
        'sr_latn', # Serbian - Latin
        'th', # Thai
        'tlh', # Klingon
        'tlh_qaak', # Klingon (pIqaD)
        'tr', # Turkish
        'uk', # Ukrainian
        'ur', #	Urdu
        'vi', #	Vietnamese
        'yua', # Yucatec Maya
        'zh_chs', # Chinese Simplified
        'zh_cht' # Chinese Traditional
    ]

    ontology_placeholder = 'n00000000'

    def __init__(self, image, credentials, labels=True, ontology=False, label_lang='en', max_labels=10, precision=2):
        self.image         = image
        self.IMAGGA_ID     = credentials['IMAGGA_ID']
        self.IMAGGA_SECRET = credentials['IMAGGA_SECRET']
        self._labels       = True
        self._ontology     = ontology
        self.label_lang    = label_lang
        self.max_labels    = max_labels
        self.precision     = precision
        self.prec_fmt      = Query.precision_fmts[precision]

    def run(self):

        auth = ( self.IMAGGA_ID, self.IMAGGA_SECRET)
        mime  = image_mime(self.image)
        fname = image_fname(self.image) 

        f = BytesFile(self.image, filename=fname)

        # 1st: POST to /content to obtain a url/content_id
        r = requests.post('https://api.imagga.com/v1/content',
                          auth=auth,
                          files={ 'image' : ( f.name, f, mime ) })
        content_id = json.loads(r.text)['uploaded'][0]['id']

        # 2nd GET from /tagging
        params = {
            'language' : self.label_lang,
            'content'  : content_id,
        }

        if self._ontology:         # the value of verbose doesn't mattter
            params['verbose'] = 1  # but its presence does. Ergo special-casing.

        r = requests.get('https://api.imagga.com/v1/tagging',
                         auth=auth, params=params)

        self._json = json.loads(r.text)


    @empty_unless('_labels')
    def labels(self):
        return self._json['results'][0]['tags']

    def tabular(self, confidence=0.0, ontology=False):
        r = []

        l_count=0
        for l in self.labels():
            if l_count == self.max_labels:
                break
            if l['confidence'] / 100. > confidence:
                l_count+=1
                if ontology:

                    # Immaga doesn't always return a synset_id.
                    # If origin == "recognition," it does
                    # If origin == "additional," it doesn't.
                    if 'synset_id' in l:
                        synset_id = l['synset_id']
                    else:
                        synset_id = self.ontology_placeholder
                  
                    r.append(self.prec_fmt.format(l['confidence'] / 100.) \
                              + '\t' + synset_id                          \
                              + '\t' + l['tag'])
                else:
                    r.append(self.prec_fmt.format(l['confidence'] / 100.) + '\t' + l['tag'])


        return r

class SightHound(Query):

  def __init__(self, image, api_token, 
               precision=2,
               faces=False, emotions=False, celebrities=False, vehicles=False, people=False):
      self.image         = image
      self.api_token     = api_token
      self.precision     = precision
      self.prec_fmt      = Query.precision_fmts[precision]

      self._faces       = faces
      self._emotions    = emotions
      self._celebrities = celebrities

      # FIXME add options for these
      self._age            = False
      self._face_landmarks = False
      self._gender         = False

      self._vehicles    = vehicles
      self._people      = people
      self._json        = {}

  def run(self):

      headers = { 'X-Access-Token' : self.api_token,
                  'Content-Type'   : 'application/octet-stream' }

      if self._faces or self._people:

          url = 'https://dev.sighthoundapi.com/v1/detections' # production / paid accounts
                                                              # use a different endpoint

          # detections?type=face,person&faceOption=gender,landmark,age,emotion
          face_opts = ''
          t  = ''

          if self._faces and self._people:
              t='face,person'

          elif self._people and not self._faces:
              t='person'

          if self._faces:
              face_opts += 'gender,'   if self._gender         else ''
              face_opts += 'landmark,' if self._face_landmarks else ''
              face_opts += 'age,'      if self._age            else ''
              face_opts += 'emotion,'  if self._emotions       else ''
              face_opts = face_opts.rstrip(',')
              if len(face_opts) > 0:
                  face_opts = '&faceOption=' + face_opts

          r = requests.post(url + '?type=' + t + face_opts,
                            headers=headers,
                            data=self.image)
          self._json['faces'] = json.loads(r.text)

      if self._vehicles:

          url = 'https://dev.sighthoundapi.com/v1/recognition?objectType=vehicle,licenseplate'
          r = requests.post(url, 
                            headers=headers, 
                            data=self.image)
          self._json['vehicles'] = json.loads(r.text)

      if self._celebrities:
          url = 'https://dev.sighthoundapi.com/v1/recognition?groupId=_celebrities'
          r = requests.post(url,
                           headers=headers,
                           data=self.image)
          self._json['celebrities'] = json.loads(r.text)

  @empty_unless('_vehicles')
  def vehicles(self):
      return self._json['vehicles']['objects']

  @empty_unless('_faces')
  def faces(self):
      return self._json['faces']['objects']

  @empty_unless('_celebrities')
  def celebrities(self):
      return self._json['celebrities']['objects']

  def tabular(self, confidence=0.0, ontology=False):
      r = []
      for vehicle in self.vehicles():

          vertices = vehicle['vehicleAnnotation']['bounding']['vertices']
          min_x = sys.maxsize
          min_y = sys.maxsize
          max_x = 0
          max_y = 0

          for vertex in vertices: # Assemble a bounding rectangle.
              max_x = vertex['x'] if vertex['x'] > max_x else max_x
              max_y = vertex['y'] if vertex['y'] > max_y else max_y
              min_x = vertex['x'] if vertex['x'] < min_x else min_x
              min_y = vertex['y'] if vertex['y'] < min_y else min_y

          #print(vehicle.keys())
          make  = vehicle['vehicleAnnotation']['attributes']['system']['make']
          model = vehicle['vehicleAnnotation']['attributes']['system']['model']
          color = vehicle['vehicleAnnotation']['attributes']['system']['color']

          r.append(str(min_x) + '\t' + str(min_y) + '\t' \
                   + str(max_x - min_x) + '\t' \
                   + str(max_y - min_y) + '\t' \
                   + self.prec_fmt.format(make['confidence']) + '\t' + make['name'] + '\t' \
                   + self.prec_fmt.format(model['confidence']) + '\t' + model['name'] + '\t' \
                   + self.prec_fmt.format(color['confidence']) + '\t' + color['name'])

      for face in self.faces():
          if face['type'] == 'face':
              box = face['boundingBox']
              r.append(str(box['x'])        + '\t' \
                       + str(box['y'])      + '\t' \
                       + str(box['width'])  + '\t' \
                       + str(box['height']))

      for celeb in self.celebrities():
          if celeb['objectType'] != 'person'                \
            or celeb['faceAnnotation']['recognitionConfidence'] < confidence:
              continue

          vertices = celeb['faceAnnotation']['bounding']['vertices']
          min_x = sys.maxsize
          min_y = sys.maxsize
          max_x = 0
          max_y = 0

          for vertex in vertices: # Assemble a bounding rectangle.
              max_x = vertex['x'] if vertex['x'] > max_x else max_x
              max_y = vertex['y'] if vertex['y'] > max_y else max_y
              min_x = vertex['x'] if vertex['x'] < min_x else min_x
              min_y = vertex['y'] if vertex['y'] < min_y else min_y

          score = celeb['faceAnnotation']['recognitionConfidence']
          name  = celeb['objectId']

          o_placeholder = self.ontology_placeholder + '\t' if ontology else ''
          r.append(str(min_x) + '\t' + str(min_y) + '\t' \
                   + str(max_x - min_x) + '\t'           \
                   + str(max_y - min_y) + '\t'           \
                   + o_placeholder                       \
                   + self.prec_fmt.format(score) + '\t'  \
                   + name)


      return r

class FacePlusPlus(Query):

    def __init__(self, image, api_key, api_secret,
                 faces=False, emotions=False,
                 precision=2):

        self.api_key    = api_key
        self.api_secret = api_secret
        self.image      = image
        
        self._faces     = faces
        self._emotions  = emotions

        self.precision  = precision
        self.prec_fmt   = Query.precision_fmts[precision]

    def run(self):
        self._json = {}
        
        form = {
            'api_key'    : self.api_key,
            'api_secret' : self.api_secret
        }
        
        mime  = image_mime(self.image)
        fname = image_fname(self.image)

        f = BytesFile(self.image, filename=fname)

        r = requests.post('https://api-us.faceplusplus.com/facepp/v3/detect',
                          data=form,
                          files={ 'image_file' : ( f.name, f, mime ) })
        self._json = json.loads(r.text)


    def tabular(self, confidence=0.0, ontology=False):
        r = []
        for face in self.faces():
            rect = face['face_rectangle']
            token = face['face_token'] # not using, for now.
            r.append(str(rect['left']) + '\t' + str(rect['top']) + '\t' \
                     + str(rect['width']) + '\t' + str(rect['height']) )

        return r

    @empty_unless('_faces')
    def faces(self):
        return self._json['faces']

    @empty_unless('_emotions')
    def emotions(self):
        raise

    
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


class Template(Query):

    def __init__(self, image, api_key, api_secret, faces=False, emotions=False,
                 precision=2):
        self.api_key    = api_key
        self.api_secret = api_secret
        self.image      = image
        self._faces     = faces
        self._emotions  = emotions
        self.precision  = precision
        self.prec_fmt   = Query.precision_fmts[precision]

    def run(self):
        raise

    def tabular(self, confidence=0.0, ontology=False):
        raise

    @empty_unless('_faces')
    def faces(self):
        raise

    @empty_unless('_emotions')
    def emotions(self):
        raise


        
if __name__ == '__main__': main()
