# Data card — ceny laptopów

## Źródło

- Dataset Kaggle z cenami laptopów (mirror campusx), zapisany w `data/raw/laptop_data.csv`.
- 1303 surowe rekordy i 12 kolumn; target: **Price** (waluta: INR, zgodnie ze zbiorem).
- Plik jest śledzony w repozytorium, żeby projekt dało się uruchomić bez pobierania danych.

Jeśli CSV nie istnieje, `data/synthetic.py` generuje deterministyczny zbiór o tym samym
schemacie cech modelowych. Dzięki temu pipeline może działać także bez realnego datasetu.

## Surowe dane → cechy modelowe

Surowy zbiór zawiera kilka pól zapisanych jako tekst. `data/features.py` parsuje je do
cech używanych przez model:

| Pole surowe | Cecha po przetworzeniu |
|---|---|
| `Ram` = `"8GB"` | `Ram` (int) |
| `Weight` = `"1.37kg"` | `Weight` (float) |
| `ScreenResolution` = `"IPS Panel ... 1920x1080"` | `Touchscreen` (0/1), `Ips` (0/1), `ppi` (float) |
| `Cpu` = `"Intel Core i5 7200U 2.5GHz"` | `Cpu_rank` (porządek: i3 < i5 < i7) |
| `Memory` = `"256GB SSD + 1TB HDD"` | `SSD` (GB), `HDD` (GB) |
| `Gpu` = `"Nvidia GeForce MX150"` | `Gpu_brand` (Intel/Nvidia/AMD) |
| `OpSys` | `Os` (Windows/Mac/Other) |

Przed tym krokiem surowy CSV przechodzi walidację Pandera (`data/contracts.py`): wymagane
są m.in. kolumny źródłowe, dodatnia przekątna ekranu i dodatni target `Price`.

Warstwa czyszczenia usuwa kolumnę indeksu oraz rekordy, dla których nie da się poprawnie
zbudować kluczowych cech `Ram`, `Weight` albo `ppi`. W praktyce oznacza to, że wartości
niemożliwe do sparsowania w tych polach nie trafiają już do preprocessingu.

## Braki danych

Braki, które pozostają po warstwie czyszczenia, są obsługiwane w preprocessingu:
cechy numeryczne są imputowane medianą, a kategoryczne najczęstszą wartością. Ten krok
chroni trening i predykcję przed pojedynczymi brakami w kolumnach dopuszczonych przez
schemat, ale nie zastępuje walidacji struktury danych wejściowych. Engineered dataframe
również przechodzi kontrakt Pandera przed treningiem.

## Schemat po przetworzeniu

Target: `Price`

- **Cechy numeryczne:** `Ram`, `Weight`, `Inches`, `ppi`, `SSD`, `HDD`, `Touchscreen`,
  `Ips`, `Cpu_rank`
- **Cechy kategoryczne:** `Company`, `TypeName`, `Gpu_brand`, `Os`

## Rozkład targetu

![Rozkład ceny](img/target_hist.png)

## Wybrana cecha numeryczna względem ceny

![RAM względem ceny](img/feature_scatter.png)

## Baseline modelu

Model bazowy to FLAML AutoML z LightGBM, ograniczeniami monotonicznymi i log-transformacją
targetu. Na 20% holdoucie uzyskuje orientacyjnie **MAE ≈ 9 400**, **RMSE ≈ 14 700** i
**R² ≈ 0.85** na oryginalnej skali ceny (około 0.88 na skali logarytmicznej).

Najważniejsze cechy według permutation importance to zwykle RAM, SSD, typ laptopa i poziom
CPU. Aktualne wartości metryk są zapisywane w `model/artifacts/metrics.json` i dostępne
przez `GET /model-info`.

## Walidacja retreningu

Retraining uruchamiany przez API używa bramek jakości z `config.yaml`. Pipeline najpierw
sprawdza schemat i minimalną liczbę rekordów, potem trenuje model w katalogu tymczasowym
i promuje nowe artefakty tylko wtedy, gdy metryki spełniają progi `min_r2` oraz `max_mae`.

## Monitoring driftu danych

Podczas treningu `model.train` zapisuje w `metrics.json` profil referencyjny danych:
statystyki cech numerycznych, odsetek braków i rozkłady najczęstszych kategorii. Endpoint
`GET /data-drift` ładuje aktualny dataset, buduje ten sam profil i porównuje go z profilem
referencyjnym. Dla cech numerycznych raport używa standaryzowanej zmiany średniej, a dla
cech kategorycznych odległości total variation między rozkładami.

Ten raport nie zastępuje walidacji schematu ani testu jakości modelu, ale pomaga przed
retreningiem zauważyć, że nowe dane mogą pochodzić z innej populacji, np. mają wyraźnie
więcej laptopów gamingowych, inną strukturę marek albo przesunięty rozkład RAM-u.
