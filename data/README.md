# Skup podataka

Korišćen je **COMtext.SR.legal** korpus (79 pravnih dokumenata, odnosno ~105k tokena), ručno anotiran po **IOB2** standardu, sa paralelnim varijantama **ekavice i ijekavice**. Korpus obuhvata širok spektar pravnih entiteta. U narednoj tabeli su prikazani entiteti i zastupljenost svakog od tipova entiteta u korpusu. Od ukupno 105470 tokena, njih 14113 (13.4%) pripada nekom imenovanom entitetu.

| Kategorija        | Potkategorija                                | Oznaka       | Broj / procenat entiteta u korpusu | Prosječna dužina u tokenima |
| ----------------- | -------------------------------------------- | ------------ | ---------------------------------- | -------------------------- |
| Osobe             |                                              | **PER**      | 694 / 19.2%                        | 1.89                       |
| Lokacije          | Lokacije/toponimi                            | **LOC**      | 294 / 8.1%                         | 1.24                       |
|                   | Adrese                                       | **ADR**      | 203 / 5.6%                         | 6.96                       |
| Organizacije      | Sudovi                                       | **COURT**    | 148 / 4.1%                         | 3.59                       |
|                   | Institucije                                  | **INST**     | 395 / 10.9%                        | 2.75                       |
|                   | Kompanije                                    | **COM**      | 337 / 9.3%                         | 2.28                       |
|                   | Ostale organizacije                          | **OTHORG**   | 97 / 2.7%                          | 2.27                       |
| Pravna dokumenta  | Opšti pravni akti                            | **LAW**      | 395 / 10.9%                        | 9.17                       |
|                   | Pojedinačni pravni akti                      | **REF**      | 227 / 6.3%                         | 10.52                      |
| Poverljivi podaci | JMBG                                         | **IDPER**    | 21 / 0.6%                          | 1.0                        |
|                   | Matični broj firme                           | **IDCOM**    | 33 / 0.9%                          | 1.0                        |
|                   | PIB                                          | **IDTAX**    | 16 / 0.4%                          | 1.0                        |
|                   | Broj računa u banci                          | **NUMACC**   | 6 / 0.2%                           | 1.0                        |
|                   | Broj lične karte/pasoša                      | **NUMDOC**   | 9 / 0.2%                           | 1.0                        |
|                   | Broj registarskih tablica/šasije             | **NUMCAR**   | 6 / 0.2%                           | 2.67                       |
|                   | Broj katastarske parcele/lista nepokretnosti | **NUMPLOT**  | 67 / 1.9%                          | 1.0                        |
|                   | Ostali ID brojevi                            | **IDOTH**    | 18 / 0.5%                          | 1.0                        |
|                   | E-mail, URL, broj telefona                   | **CONTACT**  | 8 / 0.2%                           | 1.0                        |
|                   | Datumi                                       | **DATE**     | 352 / 9.7%                         | 4.1                        |
|                   | Novčani iznosi                               | **MONEY**    | 246 / 6.8%                         | 2.32                       |
| Ostalo            |                                              | **MISC**     | 39 / 1.1%                          | 5.0                        |

## Dostupnost podataka

Skup podataka **COMtext.SR.legal** dostupan je u okviru direktorijuma [`data/`](data/).

Anotirani korpusi u **CoNLL-U** formatu:

- [`data/comtext.sr.legal.ekavica.conllu`](data/comtext.sr.legal.ekavica.conllu)
- [`data/comtext.sr.legal.ijekavica.conllu`](data/comtext.sr.legal.ijekavica.conllu)

Takođe, skup je dostupan i na Hugging Face platformi - [gradientflow/legal-ner](https://huggingface.co/datasets/gradientflow/legal-ner)
