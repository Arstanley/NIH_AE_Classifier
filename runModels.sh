#!/bin/bash 
for value in {0..20} 
do 
        CUDA_VISIBLE_DEVICES=1 python3 ./ae_train_classifier.py --model_epoch=$value 
done 
echo ALL done

