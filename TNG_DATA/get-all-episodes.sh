#!/bin/bash

curl -s http://www.chakoteya.net/NextGen/episodes.htm | grep 'a href="[0-9]*.htm' | LC_ALL=C sed 's/.*a href="\([0-9]*\).htm.*/\1/' | while read EPISODE; do
    echo "checking episode #$EPISODE..."
    [ -s $EPISODE.htm ] || (
        echo "downloading $EPISODE.htm" && curl -s "http://www.chakoteya.net/NextGen/$EPISODE.htm"  >$EPISODE.tmp
        head -n $(($(wc -l $EPISODE.tmp | awk '{print $1}') - 10)) $EPISODE.tmp >$EPISODE.htm
        rm $EPISODE.tmp
    )
done
