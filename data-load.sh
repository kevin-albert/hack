#!/bin/bash

cd $(dirname $0)

function get_file() {
    FILE="_DATA/$1"
    URL=$2
    [ -s $FILE ] || (
        echo "downloading $1" && curl -m5 -s "$URL"  >html.tmp
        iconv -f windows-1252 -t utf-8 html.tmp >$FILE || echo "Unable to decode"
        rm html.tmp
    )
}

function get_series() {
    SERIES=$1
    echo "Loading Series $SERIES"
    BASE_URL="http://www.chakoteya.net/$SERIES"
    LISTING="episodes.htm"
    [ "$SERIES" = Voyager ] && LISTING="episode_listing.htm"
    curl -s -m10 "$BASE_URL/$LISTING" \
            | grep 'a href="[0-9]*.htm' \
            | LC_ALL=C sed 's/.*a href="\([0-9]*\).htm.*/\1/' \
            | while read EPISODE; do
        echo "checking episode $SERIES - $EPISODE..."
        FILE="${SERIES}_$EPISODE.html"
        URL="$BASE_URL/$EPISODE.htm"
        get_file "$FILE" "$URL"
    done
}

function get_movie() {
    MOVIE=$1 
    echo "Loading Movie $MOVIE"
    get_file $MOVIE "http://www.chakoteya.net/movies/$MOVIE.html"
}

get_series StarTrek
get_series NextGen
get_series DS9
get_series Voyager
#get_series Enterprise

get_movie movie1.html
get_movie movie2.html
get_movie movie3.html
get_movie movie4.html
get_movie movie5.html
get_movie movie6.html
get_movie movie7.html
get_movie movie8.html
get_movie movie9.html
get_movie movie10.html

