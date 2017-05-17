#!/bin/sh

EXP_PATH=experiments
READER_PATH=reader.pkl

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
	*)
            # unknown option
	    ;;
    esac
    shift # past argument or value
done

for d in $EXP_PATH/*0; do
    python -m src.classifier $d --reader_path $READER_PATH --generated_path $d/generated/ --max_words_train "$(basename $d)"
done
