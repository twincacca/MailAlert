dd=$(env TZ=Europe/Berlin date | sed 's/ /-/g')
rm out-*
wget https://www.spalv.ch/it/animali/smarriti
egrep -i "muralto|minusio|tenero|locarno" smarriti* --color | sed 's/ /\n/g' | grep href | grep -v "><img" > out-cani-gatti-$dd.txt

