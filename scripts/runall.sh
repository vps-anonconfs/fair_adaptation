#!/usr/bin/bash

# https://stackoverflow.com/questions/6593531/running-a-limited-number-of-child-processes-in-parallel-in-bash
MAX_JOBS=5
function waitforjobs {
   while [ `jobs | wc -l` -ge $MAX_JOBS ]
   do
      sleep 5
   done
}

for g in 1 2 3 14 15 18 4 20 21;
do
  waitforjobs;
  name=german_expt"$g"
  echo $name
  python scripts/trainer.py --name $name 2>&1 > logs/"$name".log &
done
wait

for g in 5 6 7 8 9 10 11 12 13 16 17 19 23 24 25 26 27 28 29 30 31 32 33 34;
do
  name=german_expt"$g"
  waitforjobs;
  echo $name
  python scripts/trainer.py --name $name 2>&1 > logs/"$name".log &
done
wait

for a in 1 2 3 14 15 18 4 20 21;
do
  waitforjobs;
  name=adult_expt"$a";
  echo $name
  python scripts/trainer.py --name $name 2>&1 > logs/"$name".log &
done
wait

for a in 5 6 7 8 9 10 11 12 13 16 17 19 23 24 25 26 27 28 29 30 31 32 33 34;
do
  waitforjobs;
  name=adult_expt"$a";
  echo $name
  python scripts/trainer.py --name $name 2>&1 > logs/"$name".log &
done
wait

for c in 1 2 3 14 15 18 4 20 21;
do
  waitforjobs;
  name=credit_expt"$c";
  echo $name
  python scripts/trainer.py --name $name 2>&1 > logs/"$name".log &
done
wait

for c in 5 6 7 8 9 10 11 12 13 16 17 19 23 24 25 26 27 28 29 30 31 32 33 34;
do
  waitforjobs;
  name=credit_expt"$c";
  echo $name
  python scripts/trainer.py --name $name 2>&1 > logs/"$name".log &
done
wait;

for df in 0.100 0.400 0.700 1.000;
do
  for e in 35 36;
  do
    waitforjobs;
    name=german_expt"$e"_"$df";
    echo $name
    python scripts/trainer.py --name $name 2>&1 > logs/"$name".log &
  done
done
wait;

for d in adult credit;
do
  for df in 0.003 0.010 0.100 0.300 0.700 1.000;
  do
    for e in 35 36;
    do
      waitforjobs;
      name="$d"_expt"$e"_"$df";
      echo $name
      python scripts/trainer.py --name $name 2>&1 > logs/"$name".log &
    done
  done
done