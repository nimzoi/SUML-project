# Spec2Price

*Inteligentna wycena laptopów — ze specyfikacji prosto do ceny.*

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

## Perspektywa biznesowo-techniczna

Projekt można traktować jako mały system wspierający wycenę sprzętu w marketplace, skupie
laptopów albo firmie refurbishingowej. Użytkownik biznesowy nie musi znać modelu: wybiera
parametry jednego laptopa albo wrzuca CSV z całą partią sprzętu i dostaje cenę, przedział
typowego błędu oraz informację, które cechy najmocniej wpłynęły na wynik. To pomaga szybko
porównać oferty, wykryć zawyżone lub zaniżone ceny i ujednolicić proces wyceny.

Od strony technicznej aplikacja jest przygotowana tak, żeby predykcja była tylko jednym
elementem cyklu życia modelu. Dane są walidowane kontraktami, model jest trenowany przez
AutoML, predykcje są wystawione przez API, a nowe dane można sprawdzić pod kątem driftu
przed retrainingiem. MLflow rejestruje eksperymenty treningowe, więc można odtworzyć, jaki
model, z jakimi metrykami i na jakich ustawieniach został użyty. Dzięki temu system jest
bliższy aplikacji ML w środowisku operacyjnym niż jednorazowemu notebookowi.

## Architektura

Projekt składa się z trzech warstw:

- **data/** — wczytuje surowy CSV i czyści dane (`features.py`: parsowanie wartości typu
  `"8GB"` → `8`, `"1.37kg"` → `1.37`, rozdzielczość → PPI/touch/IPS, CPU/pamięć/GPU →
  cechy modelowe). Gdy pliku CSV nie ma, generuje deterministyczne dane syntetyczne o tym
  samym schemacie. Kontrakty dataframe'ów są walidowane przez Pandera, a `monitoring.py`
  buduje profil danych używany do wykrywania driftu po podmianie datasetu.
- **model/** — trenuje model przez FLAML, używa log-transformacji targetu i ograniczeń
  monotonicznych, a następnie zapisuje jeden `Pipeline` scikit-learn (`model.joblib`) oraz
  metryki (`metrics.json`) z profilem danych referencyjnych. `model/retraining.py` obsługuje
  staged retraining: trening do katalogu tymczasowego, walidację jakości i dopiero potem
  promocję artefaktów. `model/tracking.py` opcjonalnie rejestruje treningi w MLflow.
- **app/** — wystawia FastAPI (`/predict`, `/explain`, `/predict-batch`, `/health`,
  `/model-info`, `/data-schema`, `/data-drift`, `/validate-data`, `/retrain`) i interfejs
  Streamlit. Kontrakty JSON są walidowane przez Pydantic. UI najpierw korzysta z API, a jeśli
  API nie odpowiada, ładuje model lokalnie. Pokazuje estymowaną cenę, przedział typowego błędu,
  wpływ cech, prosty scenariusz zmiany RAM-u, wsadową wycenę CSV oraz monitoring driftu danych.

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
├── .github/workflows/ci.yml # CI/CD: lint, testy, walidacja configów, publikacja obrazu
├── config.py                # typowany loader konfiguracji (Pydantic)
├── data/
│   ├── raw/laptop_data.csv  # śledzony dataset
│   ├── contracts.py         # Pandera: schemat raw CSV i engineered dataframe
│   ├── features.py          # surowe dane -> cechy modelowe
│   ├── monitoring.py        # profil danych + raport driftu
│   ├── synthetic.py         # generator danych syntetycznych
│   ├── load.py              # wybór real CSV / synthetic + walidacja
│   └── prepare.py           # split + ColumnTransformer
├── model/
│   ├── train.py             # FLAML + log-target -> artefakty modelu
│   ├── retraining.py        # staged retraining + walidacja + promocja artefaktów
│   ├── tracking.py          # opcjonalny MLflow tracking eksperymentów
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

## Mapa usług i portów

| Usługa | Adres | Rola w procesie |
|---|---|---|
| Streamlit UI | http://localhost:8501 | Panel dla osoby operacyjnej: pojedyncza wycena, wycena partii CSV, eksport wyników, interpretacja i monitoring |
| FastAPI | http://localhost:8000 | Warstwa integracyjna: pozwala podłączyć model do marketplace'u, CRM, wewnętrznego panelu lub procesu batch |
| Swagger UI | http://localhost:8000/docs | Miejsce dla osoby technicznej: testowanie requestów, sprawdzanie walidacji i kontraktów danych |
| ReDoc | http://localhost:8000/redoc | Czytelny opis kontraktów API do przeglądu bez klikania endpointów |
| MLflow UI | http://127.0.0.1:5000 | Panel dla właściciela modelu: historia treningów, metryki, parametry AutoML i artefakty |

W praktycznym scenariuszu pracownik skupu lub marketplace'u korzysta z UI, system zewnętrzny
może wołać FastAPI, a osoba odpowiedzialna za model sprawdza jakość danych, drift i historię
eksperymentów. Te trzy warstwy pokazują ten sam model z trzech perspektyw: biznesowej,
integracyjnej i MLOps.

## Uruchomienie lokalne bez Dockera

**Windows (goły, bez `make`) — najprościej:**

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1   # instalacja (raz, tworzy .venv)
powershell -ExecutionPolicy Bypass -File .\run.ps1     # uruchomienie UI -> http://localhost:8501
```

**Z `make` (wygoda dewelopera, jeśli jest zainstalowany):**

```bash
make install               # tworzy .venv i instaluje wszystko (runtime + dev)
                           # Windows bez make: powershell -ExecutionPolicy Bypass -File .\setup.ps1
make train                 # opcjonalnie: artefakt modelu jest już w repo
make api                   # albo: uvicorn app.api:app --host 0.0.0.0 --port 8000
make ui                    # albo: streamlit run app/ui.py
```

Cele `make` używają `.venv` automatycznie, więc nie trzeba ręcznie aktywować środowiska.
Bez `make` użyj skryptów `setup.ps1` + `run.ps1` (sekcja wyżej) albo pełnych komend po `albo:`.

## Konfiguracja

`config.yaml` jest walidowany przy starcie aplikacji. Najważniejsze sekcje:

- `data` — ścieżka do datasetu, fallback syntetyczny, target, lista cech, split testowy.
- `model` — zadanie AutoML, budżet czasu, metryka, lista estymatorów, ensemble,
  `monotone_increasing`, `log_target`, ścieżki artefaktów i seed.
- `validation` — bramki jakości dla retrainingu: minimalna liczba rekordów, minimalne R²
  i maksymalny MAE wymagane przed promocją artefaktu.
- `tracking` — opcjonalny MLflow tracking: lokalny katalog `mlruns`, nazwa eksperymentu
  i przełącznik `enabled`.
- `api` / `ui` — host, port i adres API używany przez UI.

## Dane i walidacja

Źródłem jest dataset Kaggle z cenami laptopów zapisany w `data/raw/laptop_data.csv`
(1303 surowe rekordy). Warstwa `data/features.py` czyści pola tekstowe i buduje cechy
modelowe. Jeśli plik CSV zostanie usunięty, `data/synthetic.py` wygeneruje deterministyczny
zbiór o tym samym schemacie.

Walidacja jest dwuwarstwowa: Pydantic sprawdza konfigurację i payloady API, a Pandera
sprawdza strukturę i zakresy wartości dataframe'ów. Surowy CSV jest walidowany przed
feature engineeringiem, a engineered dataframe przed treningiem. Szczegóły są w
[docs/data_card.md](docs/data_card.md).

Przy treningu zapisywany jest też profil danych referencyjnych: średnie i odchylenia cech
numerycznych oraz rozkłady najważniejszych kategorii. `GET /data-drift` porównuje ten profil
z aktualnym datasetem, więc przed retreningiem można szybko sprawdzić, czy nowe dane wyglądają
jak ta sama populacja.

## Retraining

Aby przeuczyć model na nowszych danych, podmień `data/raw/laptop_data.csv` na plik z tymi
samymi kolumnami i uruchom:

```bash
python -m model.train
```

`model/artifacts/model.joblib` i `model/artifacts/metrics.json` zostaną przebudowane.
Aplikacja użyje nowych artefaktów po restarcie.

Ten sam proces jest dostępny przez API. `POST /retrain` uruchamia job retrainingu w tle:
najpierw waliduje dane, potem trenuje model w katalogu tymczasowym, sprawdza bramki jakości
z sekcji `validation`, a dopiero po sukcesie podmienia `model.joblib` i `metrics.json`.
Status joba jest dostępny pod `GET /retrain/{job_id}`. Na środowisku publicznym ustaw
zmienną `RETRAIN_API_KEY`; wtedy `POST /retrain` wymaga nagłówka `X-API-Key`.

## Eksperymenty MLflow

Każdy trening może zostać zarejestrowany w MLflow. Domyślnie runy zapisują się lokalnie w
`mlruns/`: logowane są parametry AutoML, metryki `MAE` / `RMSE` / `R²`, tagi z typem danych
i najlepszym estymatorem oraz artefakt `model.joblib`. Aplikacja nie wymaga uruchomionego
serwera MLflow do predykcji; tracking jest dodatkiem do audytu eksperymentów.

Zakładka **Models** (Model Registry) jest celowo pusta: lokalny backend plikowy (`mlruns/`)
obsługuje wyłącznie *tracking* (parametry, metryki, artefakty). Rejestr modeli wymagałby
backendu bazodanowego (np. `sqlite`), co jest poza zakresem tego projektu — wytrenowany model
zapisujemy jako artefakt `model.joblib`, a nie jako wpis w rejestrze.

```bash
python -m model.train
make mlflow     # albo: MLFLOW_ALLOW_FILE_STORE=true mlflow ui --backend-store-uri mlruns --host 127.0.0.1 --port 5000
```

Panel MLflow jest wtedy dostępny pod http://127.0.0.1:5000. W wariancie Docker panel startuje
automatycznie razem z API i UI (`docker compose up`) jako usługa `mlflow`, również pod
http://127.0.0.1:5000 — nie jest to więc osobny, ręcznie uruchamiany serwis. Tracking można
wyłączyć w `config.yaml` przez `tracking.mlflow.enabled: false`.

## API

Swagger UI jest dostępny pod http://localhost:8000/docs, a ReDoc pod
http://localhost:8000/redoc.

Wszystkie requesty predykcyjne są walidowane przez Pydantic. Błędny payload zwraca HTTP 422,
brak modelu zwraca HTTP 503, a retraining może zwrócić HTTP 409, jeśli inny job już działa.

`POST /predict`

```json
{ "company": "Dell", "type_name": "Notebook", "inches": 15.6, "ram_gb": 8,
  "weight_kg": 1.6, "touchscreen": 0, "ips": 1, "ppi": 141.2,
  "cpu_brand": "Intel Core i5", "ssd_gb": 256, "hdd_gb": 0,
  "gpu_brand": "Intel", "os": "Windows" }
```

Przykładowa odpowiedź: `{ "price": 55000.0 }` (INR).

| Endpoint | Rola biznesowo-operacyjna | Wynik techniczny |
|---|---|---|
| `GET /health` | monitoring dostępności usługi | `status`, informacja czy model jest załadowany |
| `POST /predict` | szybka wycena jednej oferty | `price` w INR |
| `POST /explain` | uzasadnienie ceny dla użytkownika lub analityka | cena, przedział błędu, wkład cech, scenariusz RAM |
| `POST /predict-batch` | wycena dostawy lub większej listy laptopów | lista cen w tej samej kolejności co wejście |
| `GET /model-info` | audyt aktualnego modelu przed decyzjami biznesowymi | MAE, RMSE, R², data treningu, feature importance, MLflow run |
| `GET /data-schema` | kontrakt dla osoby przygotowującej nowe dane | wymagane kolumny CSV, cechy modelowe, progi walidacji |
| `GET /data-drift` | kontrola, czy nowe dane nadal przypominają dane treningowe | raport driftu względem profilu z ostatniego treningu |
| `POST /validate-data` | szybka kontrola datasetu przed kosztownym treningiem | raport Pandera/Pydantic o poprawności danych |
| `POST /retrain` | bezpieczne odświeżenie modelu na nowych danych | `job_id`; artefakty są promowane dopiero po bramkach jakości |
| `GET /retrain/{job_id}` | śledzenie procesu retrainingu | status joba, walidacja danych, walidacja modelu, metryki |

`POST /predict-batch` używa tego samego kontraktu pojedynczego laptopa, tylko opakowanego
w `{"items": [...]}`. Ten sam workflow jest dostępny w UI jako upload CSV z eksportem wyników.
`POST /explain` jest endpointem demonstracyjnym dla interpretowalności: pokazuje nie tylko
cenę, ale też dlaczego model podbił lub obniżył wycenę względem laptopa bazowego.

## Testy i jakość

```bash
make test     # pytest
make lint     # pylint >= 8
make format   # black + isort
make validate # walidacja configu i aktualnego datasetu
make mlflow   # lokalny panel eksperymentów MLflow
```

CI uruchamia walidację plików konfiguracyjnych, `pylint --fail-under=8` i `pytest` przy
pushu oraz pull requeście. Po pushu do `main` workflow buduje obraz Dockera i publikuje go
do GitHub Container Registry jako `ghcr.io/<owner>/laptop-price`.

## Streamlit Community Cloud

`app/ui.py` może działać bez osobnego API: gdy endpoint nie odpowiada, UI ładuje model
bezpośrednio i w razie potrzeby trenuje go raz przy pierwszym uruchomieniu. Do wdrożenia
w Streamlit Community Cloud wybierz repozytorium `nimzoi/SUML-project`, gałąź `main` i
plik startowy `app/ui.py`. Zależności pobierają się z `requirements.txt`, a `packages.txt`
instaluje `libgomp1` wymagane przez LightGBM.

## Dokumentacja

- [docs/INSTRUKCJA.md](docs/INSTRUKCJA.md) — instrukcja instalacji, uruchomienia i retrainingu.
- [docs/data_card.md](docs/data_card.md) — opis danych, czyszczenia, cech i baseline'u modelu.
