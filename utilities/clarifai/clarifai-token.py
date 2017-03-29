#!/usr/bin/env python3

import os
import clarifai
from clarifai.rest import ApiClient as app

def main():
    a = app(app_id=os.environ['CLARIFAI_CLIENT_ID'],
            app_secret=os.environ['CLARIFAI_CLIENT_SECRET'])
    print(a.get_token()['access_token'])

if __name__ == '__main__': main()
