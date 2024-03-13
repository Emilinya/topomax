DESIGNS="cantilever short_cantilever bridge diffuser pipe_bend twin_pipe"
NS="40 80 160 320"

echo "Warning: running this script takes a very long time and requires more" \
"than 16GB of ram. It is included just to show which commands we ran to get our results." \

read -p "Do you truly want to run this script? (yN) " yn
case ${yn:0:1} in
    y|Y|yes|Yes )
    ;;
    * )
        exit
    ;;
esac


run()
{
for METHOD in "FEM" "DEM"
do
    for DESIGN in $DESIGNS
    do
        for N in $NS
        do
            echo "#############################################################################################"
            echo "                                   " $METHOD $DESIGN $N
            echo "#############################################################################################"
            if [ "$METHOD" == "FEM" ]
            then
                python3 run.py "designs/$DESIGN.json" $N
            else
                python3 run.py "designs/$DESIGN.json" $N "-n"
            fi

            if [ $? -ne 0 ]; then
                echo "Got an error!"
                return
            fi

            echo
            python3 plot.py -m $METHOD -d $DESIGN -N $N -s
        done
    done
done
python3 -m misc.get_consistent_objectives
}
run
