# Napišite program koji od korisnika zahtijeva unos radnih sati te koliko je placen ´
# po radnom satu. Koristite ugradenu Python metodu ¯ input(). Nakon toga izracunajte koliko ˇ
# je korisnik zaradio i ispišite na ekran. Na kraju prepravite rješenje na nacin da ukupni iznos ˇ
# izracunavate u zasebnoj funkciji naziva total_euro

workHours = int(input("Unesite radne sate: "))
hourlyRate = float(input("Unesite satnicu: "))

# earnings = float(workHours * hourlyRate)
# print(earnings)

def total_euro(workHours,hourlyRate):
    earnings = float(workHours * hourlyRate)
    return earnings

earnings = total_euro(workHours,hourlyRate)
print("Radni sati: %d h" % workHours) 
print("eura/h: %.2f" % hourlyRate)
print("Ukupno: %.2f eura" % earnings)
