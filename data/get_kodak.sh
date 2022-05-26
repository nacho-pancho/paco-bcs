#!/bin/bash
wget -c --quiet http://iie.fing.edu.uy/~nacho/data/images/kodak_color.7z
wget -c --quiet http://iie.fing.edu.uy/~nacho/data/images/kodak_gray.7z
7zr x -aos kodak_color.7z
7zr x -aos kodak_gray.7z

