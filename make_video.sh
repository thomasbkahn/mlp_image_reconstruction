#!/usr/bin/bash

# recommended (tested): output (arg 2) should end with mp4
# -vf component is to pad with black pixels so width and height are both divisible by 2 (required for x264)
# vf from https://stackoverflow.com/a/20848224
ffmpeg -framerate 20 -i $1/checkpoint_%05d.png -c:v libx264 -pix_fmt yuv420p -crf 12 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" $2
