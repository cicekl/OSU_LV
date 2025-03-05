# Napišite Python skriptu koja ce u ´ citati tekstualnu datoteku naziva ˇ SMSSpamCollection.txt
# Ova datoteka sadrži 5574 SMS poruka pri cemu su neke ozna ˇ cene kao ˇ spam, a neke kao ham.

# a) Izracunajte koliki je prosje ˇ can broj rije ˇ ci u SMS porukama koje su tipa ham, a koliko je ˇ
# prosjecan broj rije ˇ ci u porukama koje su tipa spam. ˇ
# b) Koliko SMS poruka koje su tipa spam završava usklicnikom ?

spamFile = open('SMSSpamCollection.txt')
smsData = spamFile.readlines()

hamCount = 0
hamWordCount = 0
spamCount = 0
spamWordCount = 0
counter = 0

for line in smsData:
    listMessage = line.split(None, 1)
    if(listMessage[0] == 'ham'):
        hamCount += 1
        hamWordCount += len(listMessage[1].split())

    elif(listMessage[0] == 'spam'):
        spamCount += 1
        spamWordCount += len(listMessage[1].split())
        if listMessage[1].strip().endswith('!'):
            counter += 1


print("Poruke koje zavrsavaju usklicnikom: %d" % counter)

if(hamCount > 0):
    print("Prosječan broj ham poruka: %d " % int(hamWordCount/hamCount))
else:
    print("Nema ham poruka.")

if(spamCount > 0):
    print("Prosječan broj spam poruka: %d" % int(spamWordCount/spamCount))
else:
    print("Nema spam poruka.")