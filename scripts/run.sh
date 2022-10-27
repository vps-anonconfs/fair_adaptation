#!/usr/bin/bash

for dataset in german credit adult;
do
    python scripts/trainer.py --name "$dataset"_expt121
    python scripts/trainer.py --name "$dataset"_expt122
    python scripts/trainer.py --name "$dataset"_expt123
done