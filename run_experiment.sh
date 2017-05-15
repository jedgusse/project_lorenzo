#!/bin/sh
set -e

##########################################################################################
# Runs the single experiment
#
#   Arguments
# ------------
#
#   AUTHORS authors to use for the experiment

AUTHORS="Augustinus Hipponensis,Hieronymus Stridonensis,Walafridus Strabo,Petrus Damianus,Bernardus Claraevallensis,Rabanus Maurus,Beda,Ambrosius Mediolanensis,Alcuinus,Anselmus Cantuariensis,Hugo de S- Victore,Hincmarus Rhemensis,Hildebertus Cenomanensis,Richardus S- Victoris,Tertullianus,Rupertus Tuitiensis,Honorius Augustodunensis,Prosper Aquitanus,Agobardus Lugdunensis,Gregorius I,Marbodus Redonensis,Isidorus Hispalensis,Ratherius Veronensis,Venantius Fortunatus,Boetius,Innocentius III,Petrus Blesensis,Bruno Astensis,Petrus Abaelardus,Philippus de Harveng,Gerhohus Reicherspergensis,Cyprianus Carthaginensis,Hucbaldus S- Amandi,Hrothsuita Gandersheimensis,Hilarius Pictaviensis,Othlonus S- Emmerammi Ratisponensis,Bernaldus Constantiensis,Reinerus S- Laurentii Leodiensis,Alanus de Insulis,Ausonius Burdigalensis,Fulgentius Ruspensis,Guillelmus abbas,Sigebertus Gemblacensis,Petrus Cluniacensis,Ennodius Ticinensis,Berno Augiae Divitis,Paschasius Radbertus,Leo I,Joannes Scotus Erigena,Lactantius"

#   EXP_PATH path to serialized reader
EXP_PATH=experiments/test
##########################################################################################

echo "Creating Reader"
python -m src.data --foreground_authors "$AUTHORS" --path $EXP_PATH/reader

echo "Training generators"
python -u -m src.generator \
	   --reader_path $EXP_PATH/reader.pkl \
    	   --save_path $EXP_PATH \
	   --generate \
	   --gpu \
	   --epochs 75 >> $EXP_PATH/train.log 2>&1

echo "Training classifier"
python -m src.classifier $EXP_PATH \
       --reader_path $EXP_PATH/reader.pkl \
       --generated_path $EXP_PATH/generated \

# ---*--- END
