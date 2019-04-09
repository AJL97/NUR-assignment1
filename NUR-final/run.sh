#!/bin/bash

echo "Running Assignment 1 template of Amy Louca (1687077)"

echo "Creating the plotting directory if it does not exist"
if [ ! -d "plots" ]; then
  echo "Directory does not exist create it!"
  mkdir plots
fi

echo "Creating the textfiles directory if it does not exist"
if [ ! -d "textfiles" ]; then
  echo "Directory does not exist create it!"
  mkdir textfiles
fi

echo "Running script Q1a.py ..."
python3 Q1a.py > textfiles/poisson.txt

echo "Running script Q1b.py ..."
python3 Q1b.py 

echo "Running script Q2a.py ..."
python3 Q2a.py > textfiles/integration.txt

echo "Running script Q2b.py ..."
python3 Q2b.py

echo "Running script Q2c.py ..."
python3 Q2c.py > textfiles/differentiation.txt

echo "Running Script Q2d.py ..."
python3 Q2d.py > textfiles/satel_coords.txt

echo "Running Script Q2e.py ..."
python3 Q2e.py 

echo "Running Script Q2f.py ..."
python3 Q2f.py > textfiles/roots.txt

echo "Running Script Q2g.py ..."
python3 Q2g.py > textfiles/percentiles.txt

echo "Running Script Q2h.py ..."
python3 Q2h.py > textfiles/3Dinterpolation.txt

echo "Running Script Q3a.py ... (<-- might take a while ...)"
python3 Q3a.py > textfiles/likelihood.txt

echo "Running Script Q3b.py ..."
python3 Q3b.py 

echo "Generating the pdf"

pdflatex assignment1.tex