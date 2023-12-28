rm old-out* diff-*
rm smarriti
# dd=$(date | sed 's/ /-/g')
dd=$(env TZ=Europe/Berlin date | sed 's/ /-/g')
#dd=$(env TZ=Europe/Berlin date +%Y%m%d  | sed 's/ /-/g')

#echo Ciaociao > out-$dd.txt
#date >> out-$dd.txt
#cat user_input.txt | cowsay >> out-$dd.txt

# check cani/gatti smarriti in zona
rename 's/out-cani-gatti-/old-out-cani-gatti-/' *
#wget http://spalv.ch/it/animaliwait
wget https://www.spalv.ch/it/animali/smarriti
egrep -i "muralto|minusio|tenero|locarno" smarriti --color | sed 's/ /\n/g' | grep href | grep -v "><img" > out-cani-gatti-$dd.txt
rm smarriti
diff old-out-cani-gatti-* out-cani-gatti-* > diff-cani-gatti-$dd.txt
#rm -r out-cani-gatti-*

# # check pigiama aldi
# rename 's/out-pigiama-aldi-/old-out-pigiama-aldi-/' *
# wget -r -l1  https://www.aldi-suisse.ch/it/promozioni/promozioni-e-offerte-attuali/
# wait
# grep -i "pigiama" -r www.aldi-suisse.ch  > out-pigiama-aldi-$dd.txt
# rm -r www.aldi-suisse.ch
# diff old-out-pigiama-aldi-* out-pigiama-aldi-* > diff-pigiama-aldi-$dd.txt
# #rm -r out-pigiama-aldi-*
#
# # check pigiama lidl
# rename 's/out-pigiama-lidl-/old-out-pigiama-lidl-/' *
# wget -r -l2  https://www.lidl.ch/
# wait
# find www.lidl.ch | grep Current_Sales_Week | egrep -i "pigiama|pijama|pyjama"> out-pigiama-lidl-$dd.txt
# rm -r www.lidl.ch
# diff old-out-pigiama-lidl-* out-pigiama-lidl-* > diff-pigiama-lidl-$dd.txt
# #rm -r out-pigiama-lidl-*

