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
	*)
            # unknown option
	    ;;
    esac
    shift # past argument or value
done

for d in $EXP_PATH/$FOLDER; do
    echo $d
    python -m src.classifier $d --reader_path $READER_PATH --generated_path $d/generated/ --max_words_train "$(basename $d)"
done
