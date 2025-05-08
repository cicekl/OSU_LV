# Napišite program koji od korisnika zahtijeva unos brojeva u beskonacnoj petlji ˇ
# sve dok korisnik ne upiše „Done“ (bez navodnika). Pri tome brojeve spremajte u listu. Nakon toga
# potrebno je ispisati koliko brojeva je korisnik unio, njihovu srednju, minimalnu i maksimalnu
# vrijednost. Sortirajte listu i ispišite je na ekran. Dodatno: osigurajte program od pogrešnog unosa
# (npr. slovo umjesto brojke) na nacin da program zanemari taj unos i ispiše odgovaraju ˇ cu poruku. 

list = []

while True:
    inputNum = input("Unesi broj: ")
    
    if inputNum == "Done":
        break
    else:
        
        try:
            num = int(inputNum)
            list.append(num)
        
        except:
            print("Unijeli ste znak/ove umjesto brojke.")
    


listLength = len(list)

sum = 0

for i in range(listLength):
    sum += list[i]
    
arithmeticMean = float(sum/listLength)

minNum = min(list)
maxNum = max(list)

print("\nBroj unesenih brojeva:")
print(listLength)
print("\n")

print("Srednja vrijednost liste:")
print(arithmeticMean)
print("\n")


print("Minimalna vrijednost liste:")
print(minNum)
print("\n")


print("Maksimalna vrijednost liste:")
print(maxNum)
print("\n")

    
print("Lista prije sortiranja:")
print(list)
print("\n")


list.sort()
print("Lista poslije sortiranja:")
print(list)