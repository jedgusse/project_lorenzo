
##########################################################################################
# Runs the full experiment for different values of max number of words per document
#
# - Arguments
#
#   AUTHORS authors to use for the experiment
AUTHORS="Augustinus Hipponensis,Hieronymus Stridonensis,Bernardus Claraevallensis,Walafridus Strabo,Rabanus Maurus,Beda,Ambrosius Mediolanensis,Alcuinus,Anselmus Cantuariensis,Hugo de S- Victore,Rupertus Tuitiensis"
#
#   MAX_WORDS array of values specifying max nb of words per document per run
MAX_WORDS=(500 1000 5000 10000 20000);
#
#   LOAD_PATH path to serialized reader
LOAD_PATH=reader
##########################################################################################

python src/data.py --foreground_authors "$AUTHORS" --path $LOAD_PATH

for MAX_WORD in "${MAX_WORDS[@]}"; do
    python -u src/generator.py --gpu --epochs 75 \
	   --foreground_authors "$AUTHORS" \
	   --crop_docs $MAX_WORD \
	   --load_data $LOAD_PATH.pkl \
	   --save_path $MAX_WORD >> experiments/$MAX_WORD.log 2>&1
    python src/classifier.py experiments/$MAX_WORD \
	   --reader_path $LOAD_PATH.pkl \
	   --save_generated \
	   --generated_path experiments/$MAX_WORD/generated >> experiments/$MAX_WORD.log 2>&1
done

# ---*--- END
