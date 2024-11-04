#!/bin/bash
# Create a directory to store the GADM shapefiles
mkdir -p data/gadm36
# Download the GADM shapefile for India
wget https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_IND_shp.zip -O data/gadm36/gadm41_IND_shp.zip
# Unzip the downloaded shapefile into the specified directory
unzip -o data/gadm36/gadm41_IND_shp.zip -d data/gadm36
# Remove the zip file after extraction to save space
rm data/gadm36/gadm41_IND_shp.zip
# List the contents of the directory to confirm successful extraction
ls -la data/gadm36/
