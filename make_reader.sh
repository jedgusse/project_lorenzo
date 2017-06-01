#!/bin/sh

AUTHORS="Augustinus Hipponensis,Hieronymus Stridonensis,Walafridus Strabo,Petrus Damianus,Bernardus Claraevallensis,Rabanus Maurus,Beda,Ambrosius Mediolanensis,Alcuinus,Anselmus Cantuariensis,Hugo de S- Victore,Hincmarus Rhemensis,Hildebertus Cenomanensis,Richardus S- Victoris,Tertullianus,Rupertus Tuitiensis,Honorius Augustodunensis,Prosper Aquitanus,Agobardus Lugdunensis,Gregorius I,Marbodus Redonensis,Isidorus Hispalensis,Ratherius Veronensis,Venantius Fortunatus,Boetius,Innocentius III,Petrus Blesensis,Bruno Astensis,Petrus Abaelardus,Philippus de Harveng,Gerhohus Reicherspergensis,Cyprianus Carthaginensis,Hucbaldus S- Amandi,Hrothsuita Gandersheimensis,Hilarius Pictaviensis,Othlonus S- Emmerammi Ratisponensis,Bernaldus Constantiensis,Reinerus S- Laurentii Leodiensis,Alanus de Insulis,Ausonius Burdigalensis,Fulgentius Ruspensis,Guillelmus abbas,Sigebertus Gemblacensis,Petrus Cluniacensis,Ennodius Ticinensis,Berno Augiae Divitis,Paschasius Radbertus,Leo I,Joannes Scotus Erigena,Lactantius"
# # AUTHORS="Augustinus Hipponensis,Hieronymus Stridonensis,Walafridus Strabo,Petrus Damianus,Bernardus Claraevallensis,Rabanus Maurus,Beda,Ambrosius Mediolanensis,Alcuinus,Anselmus Cantuariensis,Hugo de S- Victore,Hincmarus Rhemensis,Hildebertus Cenomanensis,Richardus S- Victoris,Tertullianus,Rupertus Tuitiensis,Honorius Augustodunensis,Prosper Aquitanus,Agobardus Lugdunensis,Gregorius I"
SEED=1000
EXP_PATH=experiments

while [[ $# -gt 1 ]]
do
    key="$1"
    case $key in
	--exp_path)
	    EXP_PATH="$2"
	    shift
	    ;;
	--seed)
	    SEED="$2"
	    shift # past argument
	    ;;
        --omega)
	    OMEGA="$2"
	    shift # past argument
	    ;;
        --alpha)
	    ALPHA="$2"
	    shift # past argument
	    ;;
	*)
            # unknown option
	    ;;
    esac
    shift # past argument or value
done

READER_PATH=$EXP_PATH/reader

if [ ! -z $OMEGA ] && [ ! -z $ALPHA ]; then
    python -m src.data --foreground_authors "$AUTHORS" --path $READER_PATH --seed $SEED --omega_path $EXP_PATH/$OMEGA --alpha_path $EXP_PATH/$ALPHA
elif [ ! -z $OMEGA ]; then
    python -m src.data --foreground_authors "$AUTHORS" --path $READER_PATH --seed $SEED --omega_path $EXP_PATH/$OMEGA
elif [ ! -z $ALPHA ]; then
    python -m src.data --foreground_authors "$AUTHORS" --path $READER_PATH --seed $SEED --alpha_path $EXP_PATH/$ALPHA
else
    python -m src.data --foreground_authors "$AUTHORS" --path $READER_PATH --seed $SEED
fi
