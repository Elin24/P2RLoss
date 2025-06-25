#!/bin/bash

datadir=/qnap/home_archive/wlin38/crowd/data/ori_data/ShanghaiTech/part_A
name=sha

part=5
T=${name}-L${part}
#T='no_mask'

mkdir exp
mkdir exp/$T
mkdir exp/$T/code
cp -r datasets exp/$T/code/datasets
cp -r models exp/$T/code/models
cp -r losses exp/$T/code/losses
cp ./*.py exp/$T/code/
cp run.sh exp/$T/code

mkdir exp/$T/train.log
python main.py --data-path $datadir \
    --label ${part} --protocol ../dac_label/$name-$part.txt \
    --batch-size 16 --tag $T 2>&1 | tee exp/$T/train.log/running.log
