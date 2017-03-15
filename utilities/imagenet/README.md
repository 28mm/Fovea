# ImageNet Utilities

These scripts help with downloading the [ImageNet](www.image-net.org/) corpus. ImageNet can be difficult to work with because:

  1. It is very large. (the fall/'11 corpus contains 14e6 images in 19e3 categories).
  2. It contains many broken links, and may point to junk placeholder images.

Rather than download the entire ImageNet corpus, these utilities download a certain number of images in each category. Some straightforward validation is done to filter out:

  1. broken URLs
  2. non-image files (i.e. not `.jpg`, `.png`, or `.gif`).
  2. the ubiquitous flickr placeholder `.png`

To initialize a local copy of ImageNet, first obtain a list of URLs from [http://image-net.org/download-imageurls](http://image-net.org/download-imageurls). Edit `imagenet-env.sh` to reflect its location. Then run `stage-imagenet.sh`. 

````bash
$ ./stage-imagenet.sh
````

To download two images from each category, run `download-imagenet.sh`, with the desired total as its argument.

````bash
$ ./download-imagenet.sh 2
````

To expand the local copy of ImageNet to include 5 images from each category, run `download-imagenet.sh` again, with the new total as its argument.

````bash
$ ./download-imagenet.sh 5
````
