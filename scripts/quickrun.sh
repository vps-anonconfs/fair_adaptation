
for d in german #credit # adult;
do
    for e in 101;
    do
      name="$d"_expt"$e";
      echo $name
      python scripts/trainer.py --name $name 2>&1 > logs/"$name".log &
    done
done

#wait
#for d in adult german credit;
#do
#    for e in 131;
#    do
#      name="$d"_expt"$e";
#      echo $name
#      python scripts/trainer.py --name $name 2>&1 > logs/"$name".log &
#    done
#done



