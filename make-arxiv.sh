basedir=$PWD
arxivdir=paper/arxiv

mkdir $arxivdir
cd $arxivdir
mkdir src
mkdir figures

cd $basedir
cp paper/preprint/apa.bst $arxivdir
cp paper/arxiv.sty $arxivdir
cp paper/main.tex $arxivdir
cp src/references.bib $arxivdir/src
cp src/*.tex $arxivdir/src
cp code/figures/*.png $arxivdir/figures
cp code/figures/*.pdf $arxivdir/figures

./generate.sh arxiv main
open $arxivdir/main.pdf

rm $arxivdir/main.log
rm $arxivdir/main.aux
rm $arxivdir/main.blg
rm $arxivdir/main.out

cd $arxivdir/../
zip -r arxiv.zip arxiv
cd $basedir
