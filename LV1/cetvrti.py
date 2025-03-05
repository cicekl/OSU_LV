# Napišite Python skriptu koja ce u ´ citati tekstualnu datoteku naziva ˇ song.txt.
# Potrebno je napraviti rjecnik koji kao klju ˇ ceve koristi sve razli ˇ cite rije ˇ ci koje se pojavljuju u ˇ
# datoteci, dok su vrijednosti jednake broju puta koliko se svaka rijec (klju ˇ c) pojavljuje u datoteci. ˇ
# Koliko je rijeci koje se pojavljuju samo jednom u datoteci? Ispišite ih.

rijecnik = {}
songFile = open('song.txt')

for line in songFile:
    line = line.rstrip()
    words = line.split()
 
    for word in words:
        rijec = word.lower()
        if rijec in rijecnik:
            rijecnik[rijec] += 1
        else:
            rijecnik[rijec] = 1

songFile.close()
 
broj = 0
for rj in rijecnik:
    if(rijecnik[rj]==1):
        broj+=1

print(broj)