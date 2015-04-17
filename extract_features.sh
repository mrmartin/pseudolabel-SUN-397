#!/bin/bash
from_dir=/media/martin/ssd-ext4/SUN_original
while read tag; do
	identify `find ${from_dir}/${tag}/*.jpg` | grep 227x227 | wc -l
	identify `find ${from_dir}/${tag}/*.jpg` | wc -l
	find ${from_dir}/${tag}/*.jpg | grep -v '\-[0-9]' | sed "s/$/ 0/g" > image_list.txt
	num_tmp=`wc -l image_list.txt | cut -f1 -d' '`
	echo "let itterations=($num_tmp+50-1)/50"
	let itterations=($num_tmp+50-1)/50
	rm -rf fc7_file

	../../Documents/caffe/build/tools/extract_features ../../Documents/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel imagenet_val.prototxt fc7 fc7_file $itterations GPU

	head -${num_tmp} fc7_file.csv > tmp
	mv tmp fc7_${tag}.csv
done < class
rm -rf fc7_file.csv fc7_file
