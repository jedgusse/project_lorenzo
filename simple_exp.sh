AUTHORS5="Augustinus Hipponensis,Hieronymus Stridonensis,Walafridus Strabo,Petrus Damianus,Bernardus Claraevallensis"
AUTHORS10="Rabanus Maurus,Beda,Ambrosius Mediolanensis,Alcuinus,Anselmus Cantuariensis"
AUTHORS15="Hugo de S- Victore,Hincmarus Rhemensis,Hildebertus Cenomanensis,Richardus S- Victoris,Tertullianus"
AUTHORS20="Rupertus Tuitiensis,Honorius Augustodunensis,Gregorius I"
AUTHORS="Augustinus Hipponensis,Hieronymus Stridonensis,Walafridus Strabo,Petrus Damianus,Bernardus Claraevallensis,Rabanus Maurus,Beda,Ambrosius Mediolanensis,Alcuinus,Anselmus Cantuariensis,Hugo de S- Victore,Hincmarus Rhemensis,Hildebertus Cenomanensis,Richardus S- Victoris,Tertullianus,Rupertus Tuitiensis,Honorius Augustodunensis,Gregorius I"
MODEL=rnn_lm
EXP_NUM=0

while [[ $# -gt 1 ]]
do
    key="$1"
    case $key in
	--model)
	    MODEL="$2"
	    shift
	    ;;
	--exp_num)
	    EXP_NUM="$2"
	    shift # past argument
	    ;;
	*)
            # unknown option
	    ;;
    esac
    shift # past argument or value
done

if [ $EXP_NUM -eq 5 ]; then
    AUTHORS=$AUTHORS5
elif [ $EXP_NUM -eq 10 ]; then
    AUTHORS=$AUTHORS10
elif [ $EXP_NUM -eq 15 ]; then
    AUTHORS=$AUTHORS15
elif [ $EXP_NUM -eq 20 ]; then
    AUTHORS=$AUTHORS20
elif [ $EXP_NUM -eq 0 ]; then
    AUTHORS=$AUTHORS
fi


CUDA_VISIBLE_DEVICES=1 python -u -m src.generator --reader_path experiments/simple/reader.pkl --save_path experiments/simple/$MODEL/ --nb_words 5000 --nb_docs 20 --generate --gpu --epochs 50 --author_selection "$AUTHORS" --model $MODEL >> experiments/simple/$MODEL/train_$EXP_NUM.log 2>&1
