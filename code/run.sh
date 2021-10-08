#!/bin/bash

while read arg

do
    python train.py $arg
    
done < arg.txt