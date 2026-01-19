# Legal NER – poređenje Transformer modela za prepoznavanje imenovanih entiteta

Projekat iz predmeta **Uvod u istraživanje podataka** koji se bavi poređenjem transformer modela za **prepoznavanje imenovanih entiteta (NER)** u pravnom domenu na srpskom jeziku.

## Pokretanje projekta lokalno

### Preduslovi
- **GPU:** Preporučuje se GPU sa najmanje **12GB VRAM** memorije. GPU sa 8GB VRAM takođe može da se koristi, ali je potrebno smanjiti batch size.
- **Python:** Python 3.11+
- **CUDA:** CUDA toolkit za GPU podršku
- **git** i **uv**

### Instalacija

1. **Kloniranje repozitorijuma:**
```bash
git clone https://github.com/neuralmaticv/lega-ner.git
cd lega-ner
```

2. **Kreiranje virtuelnog okruženja i instalacija zavisnosti:**
```bash
uv sync
```

Ova komanda će automatski kreirati virtuelno okruženje i instalirati sve potrebne pakete definisane u `pyproject.toml` fajlu.

### Treniranje modela
Za pokretanje treniranja modela:
```bash
uv run python src/leganer_fine_tuning.py
```
Postoji mogućnost definisanja načina treniranja, koristeći argument `--mode`:
```bash
uv run python src/leganer_fine_tuning.py --mode cv
```
Dostupni načini su:
- `standard` (default): Standardno treniranje - fiksni skupovi za treniranje i evaluaciju.
- `cv`: Kros-validacija sa 10 foldova.


### Napomena o memoriji
Ako imate GPU sa 8GB VRAM, smanjite `train_batch_size` i `eval_batch_size` u konfiguraciji parametara kako biste izbjegli greške vezane za nedostatak memorije.

## Podaci
Za više informacija o skupu podataka korišćenom u ovom projektu, pogledajte [data/README.md](data/README.md).

## Rezultati
Rezultati poređenja različitih transformer modela za NER zadatak su dostupni na sljedećem linku [outputs/results/README.md](outputs/results/README.md).

## Licenca
[MIT License](LICENSE)
