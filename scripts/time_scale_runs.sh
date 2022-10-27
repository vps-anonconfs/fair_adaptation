
for d in adult credit german;
do
    for depth in 1
    do
        for w in 8 12 16 32 64 256 1024 2048
        do
            for e in 118;
            do
                name="$d"_expt"$e"_"$df";
                echo $name
                python scripts/trainer.py --name $name 2>&1 > logs/"$name".log &
            done
        done
    done
    wait
done

