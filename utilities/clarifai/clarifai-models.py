#!/usr/bin/env python3

import os
import clarifai
from clarifai.rest import ClarifaiApp as app

def main():
    a = app(app_id=os.environ['CLARIFAI_CLIENT_ID'],
            app_secret=os.environ['CLARIFAI_CLIENT_SECRET'])
    models = a.models.get_all()
    for m in models:
        info = m.get_info()
        model = info['model']
        print(model['name'] + '\t' \
              + model['id'] + '\t' \
              + model['created_at'])

if __name__ == '__main__': main()

