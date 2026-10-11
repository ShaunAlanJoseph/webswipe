#!/bin/sh
set -eu
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode -halt-on-error elsarticle-template.tex
bibtex elsarticle-template
pdflatex -interaction=nonstopmode -halt-on-error elsarticle-template.tex
pdflatex -interaction=nonstopmode -halt-on-error elsarticle-template.tex
cp elsarticle-template.pdf WebSwipe_Project_Documentation.pdf
pdflatex -interaction=nonstopmode -halt-on-error WebSwipe_Reproducibility_Supplement.tex
pdflatex -interaction=nonstopmode -halt-on-error WebSwipe_Reproducibility_Supplement.tex
