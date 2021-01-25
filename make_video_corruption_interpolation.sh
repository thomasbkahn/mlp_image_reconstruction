#!/usr/bin/bash

# recommended (tested): output (arg 2) should end with mp4
# -vf component is to pad with black pixels so width and height are both divisible by 2 (required for x264)
# vf from https://stackoverflow.com/a/20848224
# ffmpeg -framerate 30 -i $1/*.png -c:v libx264 -pix_fmt yuv420p -crf 12 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" $2

# arg 2 should end in .mkv for first case
ffmpeg -framerate 10 -pattern_type glob -i "$1/*.png" -c:v copy $2
# ffmpeg -framerate 30 -i $1/*.png -c:v libx264 -pix_fmt yuv420p -crf 12 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" $2
