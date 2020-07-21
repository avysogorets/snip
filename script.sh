#!/bin/bash

for s in 0 1 2 3 4
  do
    echo "sample ${s}" 
    python3 main.py --sample=${s} --architecture=lenet5 --data=cifar10 --pruner=base --save=1;
  done;
