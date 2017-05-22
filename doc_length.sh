#!/bin/sh
set -e

##########################################################################################
# Runs the full experiment for different values of max number of words per document
#
#    Arguments
# -------------
#
#   AUTHORS authors to use for the experiment
AUTHORS="Augustinus Hipponensis,Hieronymus Stridonensis,Walafridus Strabo,Petrus Damianus,Bernardus Claraevallensis,Rabanus Maurus,Beda,Ambrosius Mediolanensis,Alcuinus,Anselmus Cantuariensis,Hugo de S- Victore,Hincmarus Rhemensis,Hildebertus Cenomanensis,Richardus S- Victoris,Tertullianus,Rupertus Tuitiensis,Honorius Augustodunensis,Prosper Aquitanus,Agobardus Lugdunensis,Gregorius I,Marbodus Redonensis,Isidorus Hispalensis,Ratherius Veronensis,Venantius Fortunatus,Boetius,Innocentius III,Petrus Blesensis,Bruno Astensis,Petrus Abaelardus,Philippus de Harveng,Gerhohus Reicherspergensis,Cyprianus Carthaginensis,Hucbaldus S- Amandi,Hrothsuita Gandersheimensis,Hilarius Pictaviensis,Othlonus S- Emmerammi Ratisponensis,Bernaldus Constantiensis,Reinerus S- Laurentii Leodiensis,Alanus de Insulis,Ausonius Burdigalensis,Fulgentius Ruspensis,Guillelmus abbas,Sigebertus Gemblacensis,Petrus Cluniacensis,Ennodius Ticinensis,Berno Augiae Divitis,Paschasius Radbertus,Leo I,Joannes Scotus Erigena,Lactantius"
#
#   MAX_WORDS array of values specifying max nb of words per document per run
MAX_WORDS=(500 1000 5000 10000 20000 100000);
#
#   READER_PATH reader path
READER_PATH=reader
#
#   EXP_PATH top experiment folder
EXP_PATH=experiments
#   MODEL
MODEL=rnn_lm
##########################################################################################

while [[ $# -gt 1 ]]
do
    key="$1"
    case $key in
	--max_words)
	    MAX_WORDS="$2"
	    shift # past argument
	    ;;
	--reader_path)
	    READER_PATH="$2"
	    shift # past argument
	    ;;
	--model)
	    MODEL="$2"
	    shift # past argument
	    ;;	
	*)
            # unknown option
	    ;;
    esac
    shift # past argument or value
done

EXP_PATH=$EXP_PATH/$MODEL

for MAX_WORD in "${MAX_WORDS[@]}"; do
    echo "Training generators with max_words $MAX_WORD"
    python -u -m src.generator \
	   --reader_path $READER_PATH.pkl \
    	   --save_path $EXP_PATH/$MAX_WORD \
	   --generate \
	   --model $MODEL \
	   --gpu \
	   --batch_size 200 \
	   --epochs 50 \
	   --max_words_train $MAX_WORD >> $EXP_PATH/$MAX_WORD.train.log 2>&1
    # echo "Training classifier"
    # python -m src.classifier $EXP_PATH/$MAX_WORD \
    # 	   --reader_path $READER_PATH.pkl \
    # 	   --generated_path $EXP_PATH/$MAX_WORD/generated \
    # 	   --max_words_train $MAX_WORD
done

# ---*--- END
