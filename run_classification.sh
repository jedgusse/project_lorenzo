#!/bin/sh

EXP_PATH=experiments
READER_PATH=reader.pkl
FOLDER=*0

while [[ $# -gt 1 ]]
do
    key="$1"
    case $key in
	--reader_path)
	    READER_PATH="$2"
	    shift # past argument
	    ;;
	--exp_path)
	    EXP_PATH="$2"
	    shift # past argument
	    ;;
	--folder)
	    FOLDER="$2"
	    shift
	    ;;
	--omega_params)
	    OMEGA_PARAMS="$2"
	    shift
	    ;;
	--alpha_params)
	    ALPHA_PARAMS="$2"
	    shift
	    ;;
	*)
            # unknown option
	    ;;
    esac
    shift # past argument or value
done

for d in $EXP_PATH/$FOLDER; do
    echo "Classifying $d"
    MAX_WORDS_TRAIN=$(basename $d)
    if [ ! -z $OMEGA_PARAMS ] && [ ! -z $ALPHA_PARAMS ]; then
	python -m src.classifier $d --reader_path $READER_PATH --generated_path $d/generated/ --max_words_train $MAX_WORDS_TRAIN --omega_params $OMEGA_PARAMS/$MAX_WORDS_TRAIN/omega_alpha --alpha_params $ALPHA_PARAMS/$MAX_WORDS_TRAIN/alpha_omega
    elif [ ! -z $OMEGA_PARAMS ]; then
	python -m src.classifier $d --reader_path $READER_PATH --generated_path $d/generated/ --max_words_train $MAX_WORDS_TRAIN --omega_params $OMEGA_PARAMS/$MAX_WORDS_TRAIN/omega_alpha
    elif [ ! -z $ALPHA_PARAMS ]; then
	python -m src.classifier $d --reader_path $READER_PATH --generated_path $d/generated/ --max_words_train $MAX_WORDS_TRAIN --alpha_params $ALPHA_PARAMS/$MAX_WORDS_TRAIN/alpha_omega
    else
	python -m src.classifier $d --reader_path $READER_PATH --generated_path $d/generated/ --max_words_train $MAX_WORDS_TRAIN
    fi
done
