if [ $# -eq 1 ]; then
    find /media/ssd-ext4/SUN_original/* -type d | sed 's/^.*\///g' | nl -v 0 | sed 's/^ *//g' | sed 's/\t/ /g' > all_classes.txt

    while read class; do
        stringarray=($class)
        echo ${stringarray[1]}

        find /media/MartinK3TB/Datasets/SUN397/SUN_original/${stringarray[1]} -type f | sed "s/$/ ${stringarray[0]}/g" | shuf > tmp
        head -$1 tmp >> train_n.txt
        head -100 tmp | tail -n +${1} | tail -n +2 >> test_n.txt
    done < all_classes.txt

    cp train_n.txt tmp
    shuf tmp > train_$1.txt
    let "vari=100-$1"
    cp test_n.txt tmp
    shuf tmp > test_$vari.txt

    rm tmp train_n.txt test_n.txt

    #cat all_images.txt | shuf > tmp
    #c=`wc -l all_images.txt | sed 's/ .*$//g'`
    #let "test=c/10"
    #head -$test tmp > test.txt
    #let "test++"
    #tail -n +$test tmp > train.txt

    #rm tmp all_images.txt

    wc -l train_$1.txt test_${vari}.txt
else
    echo "$0 requires 1 parameter, the number of images for training per class (up to 100). The size of the test class = 100 - that"
fi

#./build/tools/caffe.bin test -model models/Places_GoogLeNet/train_val_googlenet.prototxt -weights models/Places_GoogLeNet/places_googlenet.caffemodel -gpu 0 2> finetuning_to_SUN_raw