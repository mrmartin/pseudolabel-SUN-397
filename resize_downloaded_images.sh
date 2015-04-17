echo "moat+water" > tmp

cd /media/martin/MartinK3TB/Datasets/SUN397

find . -mindepth 2 -maxdepth 4 -type d > SUN_397_categories
grep -v SUN_6_scenes SUN_397_categories > tmp
rm counts
while read l; do find $l -maxdepth 1 | grep sun_ | wc -l >> counts; done < tmp
paste counts tmp | grep -v "^0" | sed 's/^.*\./\./g' > SUN_397_categories

while read tag; do
	dest=`echo ${tag} | sed 's/^....//g' | sed 's/_/+/g' | sed 's/\//+/g'`
        mkdir /media/martin/ssd-ext4/SUN_original/$dest

	ls ${tag}/*.jpg > cur_images

	COUNTER=1;
	while read im; do
		echo "${tag}: $COUNTER: $im"
		convert "$im" -resize "227x227^" -gravity center -crop "227x227+0+0" /media/martin/ssd-ext4/SUN_original/${dest}/${COUNTER}.jpg
		let "COUNTER=$COUNTER+1"
	done < cur_images
	echo "done ${tag}"
done < SUN_397_categories

