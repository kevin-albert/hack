#!/bin/bash

for F in *.htm; do
    (iconv -f utf-8 -t ascii//TRANSLIT -c $F >$F.tmp && mv $F.tmp $F) || echo "FAIL: $F"
done
