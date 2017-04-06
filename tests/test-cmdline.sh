#!/bin/bash

label_providers="google microsoft amazon clarifai imagga watson"
face_providers="google microsoft amazon clarifai imagga watson face++ sighthound opencv"
celebrity_providers="microsoft clarifai sighthound watson"

IMG=obamas.jpg

echo "[1] Checking --labels" >&2

OKCOUNT=0
ERRCOUNT=0

for p in $label_providers; do
    
    fovea --provider $p --labels "$IMG" 2> /dev/null |
	awk ' BEGIN { l=0 ; status=0 ; }
              // { l++ ; }
              $1 !~ "[0-9].[0-9]*" { status=2 ; }
              $2 !~ "[0-9]*" { status=2 ; }
              END { if(l>0) 
                       exit status;
                    else
                       exit 3; }
            '
    if [ $? == 0 ]; then
	echo "   " SUCCESS "$p" >&2
	let OKCOUNT+=1
    else
	echo "   " FAILURE "$p" >&2
	let ERRCOUNT+=1
    fi
    
done


echo "[2] Checking --faces" >&2
for p in $face_providers; do
    
    fovea --provider $p --faces "$IMG" 2> /dev/null |
	awk ' BEGIN { l=0 ; status=0 ; }
              // { l++ ; }
              $1 !~ "[0-9]*" { status=2 ; }
              $2 !~ "[0-9]*" { status=2 ; }
              $3 !~ "[0-9]*" { status=2 ; }
              $4 !~ "[0-9]*" { status=2 ; }
              END { if(l>0) 
                       exit status;
                    else
                       exit 3; }
            '
    if [ $? == 0 ]; then
	echo "   " SUCCESS "$p" >&2
	let OKCOUNT+=1
    else
	echo "   " FAILURE "$p" >&2
	let ERRCOUNT+=1
    fi
    
done

echo "[3] Checking --celebrities" >&2
for p in $celebrity_providers; do
    
    fovea --provider $p --celebrities "$IMG" 2> /dev/null |
	awk ' BEGIN { l=0 ; status=0 ; }
              // { l++ ; }
              $1 !~ "[0-9]*" { status=2 ; }
              $2 !~ "[0-9]*" { status=2 ; }
              $3 !~ "[0-9]*" { status=2 ; }
              $4 !~ "[0-9]*" { status=2 ; }
              END { if(l>0) 
                       exit status;
                    else
                       exit 3; }
            '
    if [ $? == 0 ]; then
	echo "   " SUCCESS "$p" >&2
	let OKCOUNT+=1
    else
	echo "   " FAILURE "$p" >&2
	let ERRCOUNT+=1
    fi
done

echo "[x] Summary" >&2
echo "    ERRORS    (TOTAL): $ERRCOUNT" >&2
echo "    SUCCESSES (TOTAL): $OKCOUNT"  >&2
