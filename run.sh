#!/bin/bash

echo "Run handin 4 Liz van der Kamp (s2135752)"


# Script that returns a txt file
echo "Run the first script ..."
python3 NURHW4LizQ1.py 

# Script that pipes output to multiple files
echo "Run the second script ..."
python3 NURHW4LizQ2.py

echo "Run the third script ..."
python3 NURHW4Q3.py


echo "Generating the pdf"

pdflatex SolutionsHW4Liz.tex
bibtex SolutionsHW4Liz.aux
pdflatex SolutionsHW4Liz.tex
pdflatex SolutionsHW4Liz.tex
