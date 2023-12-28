rm old-out* diff-*

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

