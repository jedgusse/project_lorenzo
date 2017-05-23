AUTHORS="Augustinus Hipponensis,Hieronymus Stridonensis,Walafridus Strabo,Petrus Damianus,Bernardus Claraevallensis,Rabanus Maurus,Beda,Ambrosius Mediolanensis,Alcuinus,Anselmus Cantuariensis,Hugo de S- Victore,Hincmarus Rhemensis,Hildebertus Cenomanensis,Richardus S- Victoris,Tertullianus,Rupertus Tuitiensis,Honorius Augustodunensis,Prosper Aquitanus,Agobardus Lugdunensis,Gregorius I,Marbodus Redonensis,Isidorus Hispalensis,Ratherius Veronensis,Venantius Fortunatus,Boetius,Innocentius III,Petrus Blesensis,Bruno Astensis,Petrus Abaelardus,Philippus de Harveng,Gerhohus Reicherspergensis,Cyprianus Carthaginensis,Hucbaldus S- Amandi,Hrothsuita Gandersheimensis,Hilarius Pictaviensis,Othlonus S- Emmerammi Ratisponensis,Bernaldus Constantiensis,Reinerus S- Laurentii Leodiensis,Alanus de Insulis,Ausonius Burdigalensis,Fulgentius Ruspensis,Guillelmus abbas,Sigebertus Gemblacensis,Petrus Cluniacensis,Ennodius Ticinensis,Berno Augiae Divitis,Paschasius Radbertus,Leo I,Joannes Scotus Erigena,Lactantius"
#AUTHORS="Augustinus Hipponensis,Hieronymus Stridonensis,Walafridus Strabo,Petrus Damianus,Bernardus Claraevallensis,Rabanus Maurus,Beda,Ambrosius Mediolanensis,Alcuinus,Anselmus Cantuariensis,Hugo de S- Victore,Hincmarus Rhemensis,Hildebertus Cenomanensis,Richardus S- Victoris,Tertullianus,Rupertus Tuitiensis,Honorius Augustodunensis,Prosper Aquitanus,Agobardus Lugdunensis,Gregorius I"

#READER_PATH=experiments/simple/reader
READER_PATH=reader

SEED=1000

python -m src.data --foreground_authors "$AUTHORS" --path $READER_PATH --seed $SEED #--omega_path experiments/simple/omega --alpha_path experiments/simple/alpha
