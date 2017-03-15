#!/bin/bash

. imagenet-env.sh

#
# A bit of error handling

[ -d "$IMGDIR" ] && {
    echo "ERROR: DIRECTORY $IMGDIR EXISTS" >&2
    exit 1
}

[ -f "$IMGLST" ] || {
    echo "ERROR: IMAGE LIST $IMGLST DOES NOT EXIST!" >&2
    exit 1
}

mkdir "$IMGDIR" || {
    echo "ERROR: CANNOT CREATE DIRECTORY $IMGDIR" >&2
    exit 1
}

#
# Stage Imagenet

echo "Creating Imagenet category directories..." >&2

cat "$IMGLST"                     \
    | cut -d'_' -f1               \
    | sort                        \
    | uniq                        \
    | while read category
do
    mkdir "${IMGDIR}/${category}"
done

echo "Adding URL manifests to category directories..." >&2
echo "(This may take a while)" >&2

sort "$IMGLST"                    \
    | while read line
do
    TARGET="${IMGDIR}/${line%%_*}/MANIFEST"
    echo "$line" >> "$TARGET"
done

