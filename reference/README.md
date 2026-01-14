# Referentni kod - COMtext.SR implementacija

Ovaj direktorijum sadrži izmijenjenu verziju originalnog koda iz [COMtext.SR projekta](https://github.com/ICEF-NLP/COMtext.SR/tree/main), koji služi kao referentna implementacija za poređenje rezultata.

## Svrha
Referentni kod koristi `simpletransformers` biblioteku (kao i originalna implementacija) kako bi se omogućilo:
- Reprodukcija originalnih rezultata iz COMtext.SR projekta
- Razumijevanje metodologije korištene u originalnoj implementaciji


## Korištenje

```bash
uv run python reference/bertic_comtext_pretokenized.py [opcije]
```

### Argumenti

| Argument | Kraće | Opis | Default |
|----------|-------|------|---------|
| `--model` | `-m` | Model: 0=BERTic, 1=SrBERTa | 0 |
| `--dialect` | `-d` | Dijalekt: 0=Ekavica, 1=Ijekavica | 0 |
| `--epochs` | `-e` | Broj epoha treniranja | 20 |


## Output direktorijumi

- **Model checkpoints**: `outputs/reference/{model}_{dialect}_{timestamp}/`

- **Rezultati (JSON)**: `results/reference/results_pretokenized_CV_{model}_{dialect}_{timestamp}.json`

<br>

---

> **Napomena**
Za originalni kod, pogledajte zvanični [GitHub repozitorijum COMtext.SR](https://github.com/ICEF-NLP/COMtext.SR/tree/main).
