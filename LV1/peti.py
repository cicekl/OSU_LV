# Napišite Python skriptu koja ce u ´ citati tekstualnu datoteku naziva ˇ SMSSpamCollection.txt
# Ova datoteka sadrži 5574 SMS poruka pri cemu su neke ozna ˇ cene kao ˇ spam, a neke kao ham.

# a) Izracunajte koliki je prosje ˇ can broj rije ˇ ci u SMS porukama koje su tipa ham, a koliko je ˇ
# prosjecan broj rije ˇ ci u porukama koje su tipa spam. ˇ
# b) Koliko SMS poruka koje su tipa spam završava usklicnikom ?

spamFile = open('../SMSSpamCollection.txt')
smsData = spamFile.readlines()

hamCount = 0
hamWordCount = 0
spamCount = 0
spamWordCount = 0
counter = 0

for line in smsData:
    label, message = line.split(None, 1)
    if(label == 'ham'):
        hamCount += 1
        hamWordCount += len(message.split())

    elif(label == 'spam'):
        spamCount += 1
        spamWordCount += len(message.split())
        if message.strip().endswith('!'):
            counter += 1


print("Poruke koje zavrsavaju usklicnikom: %d" % counter)

if(hamCount > 0):
    print("Prosječan broj ham poruka.", int(hamWordCount/hamCount))
else:
    print("Nema ham poruka.")

if(spamCount > 0):
    print("Prosječan broj spam poruka.", int(spamWordCount/spamCount))
else:
    print("Nema spam poruka.")