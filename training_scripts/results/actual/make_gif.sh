#!/bin/bash
# Install imagemagick if not there
# sudo apt-get install imagemagick
# need to expand the memory
convert -scale 100% -delay 10 -loop 0 t=*.png animation-100.gif

