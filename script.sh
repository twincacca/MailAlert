rm old-out* diff-*
dd=$(env TZ=Europe/Berlin date | sed 's/ /-/g')

#-------- rename out-cani-gatti-* ---------
ls out-cani-gatti-* | awk '{print "mv " $1 " old-"$1}' > run.sh
bash run.sh
rm run.sh
#-------- rename out-cani-gatti-* ---------END

wget https://www.spalv.ch/it/animali/smarriti
egrep -i "muralto|minusio|tenero|locarno" smarriti* --color | sed 's/ /\n/g' | grep href | grep -v "><img" > out-cani-gatti-$dd.txt
rm animali*
touch diff
