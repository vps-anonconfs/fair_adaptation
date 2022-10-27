

# https://stackoverflow.com/questions/6593531/running-a-limited-number-of-child-processes-in-parallel-in-bash
MAX_JOBS=5
function waitforjobs {
   while [ `jobs | wc -l` -ge $MAX_JOBS ]
   do
      sleep 5
   done
}

#for d in german credit adult;
#do
#    for e in 100;
#    do
#      waitforjobs;
#      name="$d"_expt"$e";
#      echo $name
#      python scripts/trainer.py --name $name 2>&1 > logs/"$name".log &
#    done
#done

# These run our figure 1 results (STD, PGD, R-IBP, F-IBP) 
# Need to recode PGD
#for d in adult credit german;
#do
#    for e in 0 100 101 102 103 110;
#    do
#      waitforjobs;
#      name="$d"_expt"$e";
#      echo $name
#      python scripts/trainer.py --name $name 2>&1 > logs/"$name".log &
#    done
#done

# These runs check if we can update a model from one to another
#for d in adult credit german;
#do
#    for e in 104 105 106 107 108 109 1101 1102;
#    do
#      waitforjobs;
#      name="$d"_expt"$e";
#      echo $name
#      python scripts/trainer.py --name $name 2>&1 > logs/"$name".log &
#    done
#done

#for d in credit;
#do
#    for e in 111 112 113;
#    do
#      waitforjobs;
#      name="$d"_expt"$e";
#      echo $name
#      python scripts/trainer.py --name $name 2>&1 > logs/"$name".log &
#    done
#done

#for d in credit;
#do
#    for e in 114 115 116;
#    do
#      waitforjobs;
#      name="$d"_expt"$e";
#      echo $name
#      python scripts/trainer.py --name $name 2>&1 > logs/"$name".log &
#    done
#done



for d in adult credit;
do
  for df in 0.050 0.200 0.300 0.700 1.000;
  do
    for e in 117;
    do
      waitforjobs;
      name="$d"_expt"$e"_"$df";
      echo $name
      python scripts/trainer.py --name $name 2>&1 > logs/"$name".log &
    done
  done
done

for d in german;
do
  for df in 0.050 0.100 0.200 0.300 0.700 1.000;
  do
    for e in 117;
    do
      waitforjobs;
      name="$d"_expt"$e"_"$df";
      echo $name
      python scripts/trainer.py --name $name 2>&1 > logs/"$name".log &
    done
  done
done



























