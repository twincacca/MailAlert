rm out*

# dd=$(date | sed 's/ /-/g')
dd=$(env TZ=Europe/Berlin date | sed 's/ /-/g')
#dd=$(env TZ=Europe/Berlin date +%Y%m%d  | sed 's/ /-/g')

#echo Ciaociao > out-$dd.txt
#date >> out-$dd.txt
#cat user_input.txt | cowsay >> out-$dd.txt

# # check cani/gatti smarriti in zona
# wget http://spalv.ch/it/animali
# wait
# egrep -i "muralto|minusio|tenero|locarno" animali --color | sed 's/ /\n/g' | grep href | grep -v "><img" > out-cani-gatti-$dd.txt
rm animali

# check pigiama aldi
# wget -r -l1  https://www.aldi-suisse.ch/it/promozioni/promozioni-e-offerte-attuali/
# wait
# grep -i "pigiama" -r www.aldi-suisse.ch  > out-pigiama-aldi-$dd.txt
rm -r www.aldi-suisse.ch

# check pigiama lidl
wget -r -l1  https://www.lidl.ch/c/it-CH/azioni-della-settimana/a10029621?tabCode=Current_Sales_Week
wait
find www.lidl.ch | grep Current_Sales_Week | egrep -i "pigiama|pijama|pyjama"> out-pigiama-lidl-$dd.txt
rm -r www.lidl.ch
