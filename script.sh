rm out*
dd=$(date | sed 's/ /-/g')
#echo Ciaociao > out-$dd.txt
#date >> out-$dd.txt
#cat user_input.txt | cowsay >> out-$dd.txt

# check cani/gatti smarriti in zona
rm animali
wget http://spalv.ch/it/animali
wait
egrep -i "muralto|minusio|locarno" animali --color | sed 's/ /\n/g' | grep href | grep -v "><img" > out-$dd.txt
