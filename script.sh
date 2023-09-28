rm out*

#dd=$(date | sed 's/ /-/g')
dd=$(env TZ=Europe/Berlin date)
#dd=$(env TZ=Europe/Berlin date +%Y%m%d)

#echo Ciaociao > out-$dd.txt
#date >> out-$dd.txt
#cat user_input.txt | cowsay >> out-$dd.txt

# check cani/gatti smarriti in zona
rm animali
wget http://spalv.ch/it/animali
wait
egrep -i "muralto|minusio|tenero|locarno" animali --color | sed 's/ /\n/g' | grep href | grep -v "><img" > out-cani-gatti-$dd.txt

# # check pigiama aldi
# rm -r www.aldi-suisse.ch
# wget -r -l1  https://www.aldi-suisse.ch/it/promozioni/promozioni-e-offerte-attuali/
# wait
# grep -i "pigiama" -r www.aldi-suisse.ch  > out-pigiama-aldi-$dd.txt
