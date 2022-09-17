#!/bin/bash
file="classic_images_grayscale.zip"
wget -c http://iie.fing.edu.uy/~nacho/data/images/${file}
unzip ${file}
file="classic_color.zip"
wget -c http://iie.fing.edu.uy/~nacho/data/images/${file}
unzip ${file}
file="misc_grayscale.zip"
wget -c http://iie.fing.edu.uy/~nacho/data/images/${file}
unzip ${file}
rm *.zip
for i in *.pgm; do convert $i ${i/pgm/png}; done
for i in *.ppm; do convert $i ${i/ppm/png}; done
