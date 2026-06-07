# Wycena laptopa

![CI](https://github.com/nimzoi/SUML-project/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Code style: black](https://img.shields.io/badge/code%20style-black-000000)
![pylint](https://img.shields.io/badge/pylint-10.00%2F10-brightgreen)

Aplikacja szacuje cenę laptopa na podstawie specyfikacji. Model regresyjny jest trenowany
przez **AutoML (FLAML)**, udostępniony przez **FastAPI** i obsługiwany z poziomu
interfejsu **Streamlit**. Projekt ma osobne warstwy `data`, `model` i `app`, a konfiguracja
jest zebrana w jednym pliku `config.yaml`.

> Kurs: Środowiska uruchomieniowe ML (SUML), PJATK — projekt grupowy.

![Interfejs wyceny laptopa](docs/img/ui.png)

## Kontekst

Poprawna wycena laptopa jest przydatna w handlu, marketplace'ach i narzędziach do
weryfikacji ofert. Model estymuje cenę na podstawie marki, typu obudowy, RAM-u, dysków,
ekranu, CPU/GPU i systemu operacyjnego. Wynik jest dostępny przez endpoint `/predict` oraz
w UI. Aktualny baseline: **R² ≈ 0.85** na oryginalnej skali ceny, MAE ≈ 9 600 INR.

## Architektura

Projekt składa się z trzech warstw:

- **data/** — wczytuje surowy CSV i czyści dane (`features.py`: parsowanie wartości typu
  `"8GB"` → `8`, `"1.37kg"` → `1.37`, rozdzielczość → PPI/touch/IPS, CPU/pamięć/GPU →
  cechy modelowe). Gdy pliku CSV nie ma, generuje deterministyczne dane syntetyczne o tym
  samym schemacie.
- **model/** — trenuje model przez FLAML, używa log-transformacji targetu i ograniczeń
  monotonicznych, a następnie zapisuje jeden `Pipeline` scikit-learn (`model.joblib`) oraz
  metryki (`metrics.json`).
- **app/** — wystawia FastAPI (`/predict`, `/health`, `/model-info`) i interfejs Streamlit.
  UI najpierw korzysta z API, a jeśli API nie odpowiada, ładuje model lokalnie. Pokazuje
  estymowaną cenę, przedział typowego błędu, wpływ cech i prosty scenariusz zmiany RAM-u.

Zmiana źródła danych lub parametrów AutoML wymaga edycji `config.yaml`, nie kodu.

## Struktura repozytorium

```text
SUML-project/
├── config.yaml              # główna konfiguracja danych, modelu i serwowania
├── requirements.txt         # zależności runtime
├── requirements-dev.txt     # zależności developerskie i testowe
├── Dockerfile               # obraz slim, użytkownik non-root, trening przy buildzie
├── docker-compose.yml       # serwisy: api (FastAPI) i ui (Streamlit)
├── Makefile · pyproject.toml · .pylintrc · packages.txt
├── .github/workflows/ci.yml # CI: pylint (>= 8) i pytest
├── config.py                # typowany loader konfiguracji (Pydantic)
├── data/
│   ├── raw/laptop_data.csv  # śledzony dataset
│   ├── features.py          # surowe dane -> cechy modelowe
│   ├── synthetic.py         # generator danych syntetycznych
│   ├── load.py              # wybór real CSV / synthetic + walidacja
│   └── prepare.py           # split + ColumnTransformer
├── model/
│   ├── train.py             # FLAML + log-target -> artefakty modelu
│   └── evaluate.py          # MAE / RMSE / R2
├── app/
│   ├── schemas.py           # modele Pydantic dla API
│   ├── inference.py         # wspólne mapowanie payloadu na predykcję
│   ├── api.py               # FastAPI
│   └── ui.py                # Streamlit
├── tests/                   # testy jednostkowe i integracyjne
└── docs/                    # instrukcja, data card, wykresy EDA i zrzut UI
```

## Wymagania

- Python 3.11+
- Docker + Docker Compose, jeśli wybierasz uruchomienie kontenerowe

Zależności są przypięte w `requirements.txt`. Dla Dockera instalują się podczas budowy
obrazu; lokalnie instaluje je `requirements-dev.txt`.

## Szybki start z Dockerem

```bash
git clone https://github.com/nimzoi/SUML-project.git
cd SUML-project
docker compose up --build
```

Po starcie:

- API: http://localhost:8000, dokumentacja OpenAPI pod `/docs`
- UI: http://localhost:8501

Dataset jest w repozytorium, a model trenuje się podczas budowy obrazu.

## Uruchomienie lokalne bez Dockera

```bash
python -m pip install -r requirements-dev.txt
python -m model.train      # opcjonalnie: artefakt modelu jest już w repo
make api                   # albo: uvicorn app.api:app --host 0.0.0.0 --port 8000
make ui                    # albo: streamlit run app/ui.py
```

Na Windowsie, jeśli nie ma `make`, użyj pełnych komend podanych po `albo:`.

## Konfiguracja

`config.yaml` jest walidowany przy starcie aplikacji. Najważniejsze sekcje:

- `data` — ścieżka do datasetu, fallback syntetyczny, target, lista cech, split testowy.
- `model` — zadanie AutoML, budżet czasu, metryka, lista estymatorów, ensemble,
  `monotone_increasing`, `log_target`, ścieżki artefaktów i seed.
- `api` / `ui` — host, port i adres API używany przez UI.

## Dane

Źródłem jest dataset Kaggle z cenami laptopów zapisany w `data/raw/laptop_data.csv`
(1303 surowe rekordy). Warstwa `data/features.py` czyści pola tekstowe i buduje cechy
modelowe. Jeśli plik CSV zostanie usunięty, `data/synthetic.py` wygeneruje deterministyczny
zbiór o tym samym schemacie. Szczegóły są w [docs/data_card.md](docs/data_card.md).

## Retraining

Aby przeuczyć model na nowszych danych, podmień `data/raw/laptop_data.csv` na plik z tymi
samymi kolumnami i uruchom:

```bash
python -m model.train
```

`model/artifacts/model.joblib` i `model/artifacts/metrics.json` zostaną przebudowane.
Aplikacja użyje nowych artefaktów po restarcie.

## API

`POST /predict`

```json
{ "company": "Dell", "type_name": "Notebook", "inches": 15.6, "ram_gb": 8,
  "weight_kg": 1.6, "touchscreen": 0, "ips": 1, "ppi": 141.2,
  "cpu_brand": "Intel Core i5", "ssd_gb": 256, "hdd_gb": 0,
  "gpu_brand": "Intel", "os": "Windows" }
```

Przykładowa odpowiedź: `{ "price": 55000.0 }` (INR).

- `GET /health` zwraca status procesu i informację, czy model jest załadowany.
- `GET /model-info` zwraca metryki, datę treningu, najlepszy estymator i ważność cech.

Niepoprawny payload kończy się HTTP 422. Brak artefaktu modelu daje HTTP 503.

## Testy i jakość

```bash
make test     # pytest
make lint     # pylint >= 8
make format   # black + isort
```

CI uruchamia `pylint --fail-under=8` i `pytest` przy pushu oraz pull requeście.

## Streamlit Community Cloud

`app/ui.py` może działać bez osobnego API: gdy endpoint nie odpowiada, UI ładuje model
bezpośrednio i w razie potrzeby trenuje go raz przy pierwszym uruchomieniu. Do wdrożenia
w Streamlit Community Cloud wybierz repozytorium `nimzoi/SUML-project`, gałąź `main` i
plik startowy `app/ui.py`. Zależności pobierają się z `requirements.txt`, a `packages.txt`
instaluje `libgomp1` wymagane przez LightGBM.

## Dokumentacja

- [docs/INSTRUKCJA.md](docs/INSTRUKCJA.md) — instrukcja instalacji, uruchomienia i retrainingu.
- [docs/data_card.md](docs/data_card.md) — opis danych, czyszczenia, cech i baseline'u modelu.
