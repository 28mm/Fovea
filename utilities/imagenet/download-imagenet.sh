#!/bin/bash

. imagenet-env.sh

export J=10
export N=${1:-5}

#
# A bit of error handling

[ -d "$IMGDIR" ] || {
    echo "ERROR: DIRECTORY $IMGDIR DOES NOT EXIST" >&2
    exit 1
}

#
# Functions definitions

# Download n images from category
function download-n {
    N="$1"
    export CATEGORY="$2"
    echo "$CATEGORY"
    cat "${IMGDIR}/${CATEGORY}/MANIFEST" \
        | urlescape                      \
        | while read line
    do
        FILE=$( echo "$line" | cut -d' ' -f1 )
        URL=$( echo "$line" | cut -d' ' -f2 )
        CATEGORY="${FILE%%_*}"
        TARGET="${IMGDIR}/${CATEGORY}/${FILE}"

        COUNT=$( ls "${IMGDIR}/${CATEGORY}" \
            | egrep -v 'MANIFEST'  \
            | egrep -v 'ERRORS'    \
            | wc -l )

        [ "$COUNT" -gt "$N" ] && break

        # Maybe we've tried this one and failed, already?
        grep "$FILE" "${IMGDIR}/${CATEGORY}/ERRORS" \
             > /dev/null 2>&1                       \
            && continue

        # Maybe we've already got this one?
        [ -f "${TARGET}"     ] && continue
        [ -f "${TARGET}.jpg" ] && continue
        [ -f "${TARGET}.png" ] && continue
        [ -f "${TARGET}.gif" ] && continue

        # Actually download the thing
        curl -s --connect-timeout 1 -o "$TARGET" "$URL" || {
            echo "$FILE" >> "${IMGDIR}/${CATEGORY}/ERRORS"
            [ -f "$TARGET" ] && rm "$TARGET"
            continue
        }

        if file "$TARGET" | grep JPEG > /dev/null
        then
            mv "$TARGET" "$TARGET.jpg"
        elif file "$TARGET" | grep PNG > /dev/null
        then
            mv "$TARGET" "$TARGET.png"

            # We need to make sure this isn't the flickr
            # placeholder .png
            if [[ "880a7a58e05d3e83797f27573bb6d35c" \
                      == $( md5 < "$TARGET.png" ) ]]
            then
                echo "$FILE" >> "${IMGDIR}/${CATEGORY}/ERRORS"
                rm "$TARGET.png"
            fi


        elif file "$TARGET" | grep GIF > /dev/null
        then
            mv "$TARGET" "$TARGET.gif"
        else
            echo "$FILE" >> "${IMGDIR}/${CATEGORY}/ERRORS"
            [ -f "$TARGET" ] && rm "$TARGET"
        fi

    done
} ; export -f download-n

# Some of the imagenet urls have unescaped parenthesis.
function urlescape {
    sed  's/(/\%28/g
          s/)/\%29/g' <&0
} ; export -f urlescape

#
# Download Imagenet

cd "$IMGDIR"
find . -type d -depth 1  \
    | sed 's/\.\///'     \
    | parallel -L 1 -j "$J" download-n "$N" {}


