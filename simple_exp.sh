AUTHORS5="Augustinus Hipponensis,Hieronymus Stridonensis,Walafridus Strabo,Petrus Damianus,Bernardus Claraevallensis"
AUTHORS10="Rabanus Maurus,Beda,Ambrosius Mediolanensis,Alcuinus,Anselmus Cantuariensis"
AUTHORS15="Hugo de S- Victore,Hincmarus Rhemensis,Hildebertus Cenomanensis,Richardus S- Victoris,Tertullianus"
AUTHORS20="Rupertus Tuitiensis,Honorius Augustodunensis,Gregorius I"
AUTHORS50="Prosper Aquitanus,Agobardus Lugdunensis,Marbodus Redonensis,Isidorus Hispalensis,Ratherius Veronensis,Venantius Fortunatus,Boetius,Innocentius III,Petrus Blesensis,Bruno Astensis,Petrus Abaelardus,Philippus de Harveng,Gerhohus Reicherspergensis,Cyprianus Carthaginensis,Hucbaldus S- Amandi,Hrothsuita Gandersheimensis,Hilarius Pictaviensis,Othlonus S- Emmerammi Ratisponensis,Bernaldus Constantiensis,Reinerus S- Laurentii Leodiensis,Alanus de Insulis,Ausonius Burdigalensis,Fulgentius Ruspensis,Guillelmus abbas,Sigebertus Gemblacensis,Petrus Cluniacensis,Ennodius Ticinensis,Berno Augiae Divitis,Paschasius Radbertus,Leo I,Joannes Scotus Erigena,Lactantius"
TOP18="Augustinus Hipponensis,Hieronymus Stridonensis,Walafridus Strabo,Petrus Damianus,Bernardus Claraevallensis,Rabanus Maurus,Beda,Ambrosius Mediolanensis,Alcuinus,Anselmus Cantuariensis,Hugo de S- Victore,Hincmarus Rhemensis,Hildebertus Cenomanensis,Richardus S- Victoris,Tertullianus,Rupertus Tuitiensis,Honorius Augustodunensis,Gregorius I"
ALL="Augustinus Hipponensis,Hieronymus Stridonensis,Walafridus Strabo,Petrus Damianus,Bernardus Claraevallensis,Rabanus Maurus,Beda,Ambrosius Mediolanensis,Alcuinus,Anselmus Cantuariensis,Hugo de S- Victore,Hincmarus Rhemensis,Hildebertus Cenomanensis,Richardus S- Victoris,Tertullianus,Rupertus Tuitiensis,Honorius Augustodunensis,Gregorius I,Prosper Aquitanus,Agobardus Lugdunensis,Marbodus Redonensis,Isidorus Hispalensis,Ratherius Veronensis,Venantius Fortunatus,Boetius,Innocentius III,Petrus Blesensis,Bruno Astensis,Petrus Abaelardus,Philippus de Harveng,Gerhohus Reicherspergensis,Cyprianus Carthaginensis,Hucbaldus S- Amandi,Hrothsuita Gandersheimensis,Hilarius Pictaviensis,Othlonus S- Emmerammi Ratisponensis,Bernaldus Constantiensis,Reinerus S- Laurentii Leodiensis,Alanus de Insulis,Ausonius Burdigalensis,Fulgentius Ruspensis,Guillelmus abbas,Sigebertus Gemblacensis,Petrus Cluniacensis,Ennodius Ticinensis,Berno Augiae Divitis,Paschasius Radbertus,Leo I,Joannes Scotus Erigena,Lactantius"

MODEL=rnn_lm
EXP_NUM=0
EXP_PATH=experiments/simple

while [[ $# -gt 1 ]]
do
    key="$1"
    case $key in
	--model)
	    MODEL="$2"
	    shift
	    ;;
	--exp_path)
	    EXP_PATH="$2"
	    shift # past argument
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
elif [ $EXP_NUM -eq 50 ]; then
    AUTHORS=$AUTHORS50
elif [ $EXP_NUM -eq 0 ]; then
    AUTHORS=$TOP18
elif [ $EXP_NUM -eq -1 ]; then
    AUTHORS=$ALL
fi

echo "$AUTHORS"

CUDA_VISIBLE_DEVICES=1 python -u -m src.generator --data_path $EXP_PATH/reader.pkl --save_path $EXP_PATH/$MODEL/ --nb_words 5000 --nb_docs 20 --generate --gpu --epochs 50 --author_selection "$AUTHORS" --model $MODEL >> $EXP_PATH/$MODEL/train_$EXP_NUM.log 2>&1
