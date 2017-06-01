#!/bin/sh
EXP_PATH=experiments
READER_PATH=reader.pkl
MAX_AUTHORS=(5 10 25)

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
	--gener_path)
	    GENER_PATH="$2"
	    shift
	    ;;
	*)
            # unknown option
	    ;;
    esac
    shift # past argument or value
done

for m in "${MAX_AUTHORS[@]}"; do
    python -m src.classifier $EXP_PATH/$m --max_authors $m --reader_path $READER_PATH --generated_path $GENER_PATH
done
