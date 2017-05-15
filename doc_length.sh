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
MAX_WORDS=(500 1000 5000 10000 20000 0);
#
#   LOAD_PATH path to serialized reader
LOAD_PATH=reader
##########################################################################################

echo "Creating Reader"
python -m src.data --foreground_authors "$AUTHORS" --path $LOAD_PATH

for MAX_WORD in "${MAX_WORDS[@]}"; do
    echo "Training generators"
    python -u -m src.generator \
	   --reader_path $LOAD_PATH.pkl \
    	   --save_path experiments/$MAX_WORD \
	   --generate \
	   # --gpu
	   --epochs 75 \
	   --max_words $MAX_WORD >> experiments/$MAX_WORD.train.log 2>&1
    echo "Training classifier"
    python -m src.classifier experiments/$MAX_WORD \
	   --reader_path $LOAD_PATH.pkl \
	   --generated_path experiments/$MAX_WORD/generated \
	   --max_words_train $MAX_WORD
done

# ---*--- END
