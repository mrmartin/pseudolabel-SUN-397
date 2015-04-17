cd /media/martin/MartinK3TB/Datasets/SUN397

find . -mindepth 2 -maxdepth 4 -type d > SUN_397_categories
grep -v SUN_6_scenes SUN_397_categories > tmp
rm counts
while read l; do find $l -maxdepth 1 | grep sun_ | wc -l >> counts; done < tmp
paste counts tmp | grep -v "^0" | sed 's/^.*\./\./g' > SUN_397_categories

cd -
cp /media/martin/MartinK3TB/Datasets/SUN397/SUN_397_categories .
#!/bin/bash
while read tag; do
	identify `find /media/martin/MartinK3TB/Datasets/SUN397/${tag}/*.jpg` | grep 227x227 | wc -l
	identify `find /media/martin/ssd-ext4/SUN_from_google/${tag}/*.jpg` | wc -l
	find /media/martin/ssd-ext4/SUN_from_google/${tag}/*.jpg | grep -v '\-[0-9]' | sed "s/$/ 0/g" > image_list.txt
	num_tmp=`wc -l image_list.txt | cut -f1 -d' '`
	echo "let itterations=($num_tmp+50-1)/50"
	let itterations=($num_tmp+50-1)/50
	rm -rf fc7_file

	../../Documents/caffe/build/tools/extract_features ../../Documents/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel imagenet_val.prototxt fc7 fc7_file $itterations GPU

	head -${num_tmp} fc7_file.csv > tmp
	mv tmp fc7_${tag}.csv
done < class
