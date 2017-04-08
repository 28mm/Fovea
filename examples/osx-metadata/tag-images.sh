#!/bin/bash
#
# tag-image.sh. Set xattr metadata on images, so that they become
# searchable based on their contents. Works with OSX/Spotlight.


force=0             # overwrite tags?
provider="clarifai" # use clarifai, google, amazon, watson, ...
confidence=0.6      # confidence threshold

while getopts "fp:c:" option
do
	case "$option" in
		f) force=1 ;;
		p) provider="$OPTARG" ;;
		c) confidence="$OPTARG" ;;
        *) exit 1 ;;
	esac
done

shift $(($OPTIND - 1))

for image in $@
do

	#
	# Check if search-related metadata are set already.
	# i.e. have we already tagged this image.
	if xattr -r "$image"                                                    \
		| egrep '(kMDItemFinderComment|_kMDItemUserTags|kMDItemOMUserTags)' \
		> /dev/null 2>&1                                                    \
		&& [ $force == 0 ]
	then
		echo "[${image}]: Tags found! Skipping!"
		continue
	fi 

	#
	# Get tags, populate a plist/xml blob with them.
	echo "[${image}]: retrieving tags from $provider"
	plist=$( cat <<EOF
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
 <plist version="1.0"><array>
 $( fovea --provider $provider "$image" --confidence $confidence \
			| sed -E 's/^[0-9]\.[0-9]*[[:space:]]+//g'           \
			| awk '{printf("<string>%s</string>\n", $0); }' )
 </array></plist>
EOF
)

	#
	# If we don't have tags, don't modify metadata.
	if ! echo "$plist" | grep string > /dev/null 2>&1
	then
		echo "[${image}]: no tags found!"
	else

		#
		# But otherwise, we need to write our plist to three places.
		for field in "kMDItemFinderComment" "_kMDItemUserTags" "kMDItemOMUserTags"
		do
			echo "[${image}]: setting ${field}"
			xattr -w "com.apple.metadata:${field}" "$plist" "$image"
		done
	fi
done