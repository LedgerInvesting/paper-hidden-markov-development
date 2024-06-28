#! /usr/bin/env bash

:'
Generate the paper from latex templates.
'

basedir=$PWD
template=$1
target=$2

if [[ -n "$template" ]]; then
    cd paper/$1
else
    cd paper/preprint/
fi

if [[ -n "$target" ]]; then
	target=$target
else
	target=main
fi 

function generate() {
    pdflatex $target.tex
    bibtex $target
    pdflatex $target.tex
    pdflatex $target.tex
}


generate
cd $basedir
echo "Complete."
