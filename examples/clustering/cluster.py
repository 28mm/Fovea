#!/usr/bin/env python3
#
# cluster.py clusters images based on the labels assigned to them by
# an image classifier. It expects a directory of image and label files.
#
# For example: 
#
# 1.jpg, 1.jpg.labels, 2.jpg, 2.jpg.labels, 3.png, 3.png.labels
#
# Each line of a label file should have two tab-separated values:
# a confidence score, and a label. 
#
# For instance:
#
# 0.999999    a high confidence label
# 0.777777    a lower confidence label 
# 
# An example workflow follows:
#
# $ mkdir images-d
# $ cd images-d
# $ cp <a bunch of images> ./
# $ for i in *.jpg; do fovea $i > ${i}.labels ; done
# $ cluster.py --kmeans --confidence 0.7 --clusters 10 ./

import numpy as np
import pandas as pd
import os
import argparse

from sklearn import feature_extraction
from sklearn.feature_extraction import DictVectorizer
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import metrics
#import mpld3

def labeler(thing):
    return thing

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--clusters', type=int, default=5, required=False)
    parser.add_argument('--confidence', type=float, default=0.0, required=False)
    parser.add_argument('directory', nargs=1)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--kmeans', action='store_true')
    group.add_argument('--dbscan', action='store_true')
    args = parser.parse_args()

    confidence = args.confidence
    n_clusters = args.clusters
    directory  = args.directory[0]
    documents = {}

    for filename in os.listdir(directory):

        if not filename.endswith('.labels'):
            continue

        with open(directory + '/' + filename, 'r') as f:
            documents[filename] = []

            for line in f.readlines():

                l_components = line.split('\t')
                conf = float(l_components[0])
                label = l_components[1][:-1]

                if conf > confidence:
                    documents[filename].append(label)
 
    v = DictVectorizer()
    dox = [ { l : 1 for l in documents[d] } for d in documents ]
    doc_names = [ d.rstrip('.labels') for d in documents ]

    X = v.fit_transform(dox)
    features = v.get_feature_names()

    if args.kmeans:
        km = KMeans(n_clusters=n_clusters)
        km.fit(X)

        # Sort cluster centers by proximity to centroid
        order_centroids = km.cluster_centers_.argsort()[:, ::-1]
        closest_labels_dict = { i : "" for i in range(n_clusters) }

        for i in range(n_clusters):
        
            for ind in order_centroids[i, :6]: #replace 6 with n words per cluster
                closest_labels_dict[i] += features[ind] + ", "
            closest_labels_dict[i] = closest_labels_dict[i].rstrip(', ')

        clusters = km.labels_.tolist()
        clusters_dict = { i : [] for i in range(n_clusters) } 

        for c in range(len(clusters)):
            clusters_dict[clusters[c]].append(doc_names[c])

        print('<html>')
        print('<body>')

        print('<style>')
        print('img { height: 75px; }')
        print('h2 { font-family: sans-serif; } ')
        print('.box { max-width: 700px; }')
        print('</style>')

        print('<div class="box">')
        for k in clusters_dict:
            print('<h2>' + str(k) + ": " +  closest_labels_dict[k] + '</h2>')
            for img in clusters_dict[k]:
                print('<img src="file://' + directory + '/' + img + '">')
        print('</div>')
        print('</body>')
        print('</html>')

    elif args.dbscan:
        raise

if __name__ == '__main__' : main()
