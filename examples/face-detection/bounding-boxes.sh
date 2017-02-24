#!/bin/bash
#
# bounding-boxes.sh: draw bounding-boxes around faces detected in a picture.
#    iterates over the set of provders supported by fovea.

INFILE=7.png # 7.png: still from /Huit Femme/.
PROVIDERS="google microsoft amazon opencv clarifai watson"
PATH="../../bin:$PATH"
source ~/cv-env.sh

for provider in $PROVIDERS
do
    # assume we have a png.
    OUTFILE="${INFILE##.png}-${provider}.png"

    {
        # the following three commands print to a single line.
        echo -n "convert $INFILE -strokewidth 0 -fill \"rgba(255, 0, 0, 0.5)\" "
        fovea --provider $provider --faces "$INFILE" \
            | awk '{  printf("-draw \"rectangle  %s,%s %s,%s\" ", $1, $2, $1+$3, $2+$4) ; }'
        echo "$OUTFILE"
    } | bash

done

