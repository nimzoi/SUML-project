# Instrukcja instalacji i uruchomienia — Spec2Price

Aplikacja szacuje cenę laptopa na podstawie jego specyfikacji. Dostępna jest jako
**strona (Streamlit)** oraz **API (FastAPI)**. Ten dokument prowadzi krok po kroku — od
zera do działającej aplikacji — bez konieczności konfigurowania czegokolwiek w systemie.

> Wskazówka: jeśli chcesz tylko zobaczyć, jak działa — wybierz **Wariant A (Docker)**.
> To jedna komenda i nie wymaga instalowania Pythona ani żadnych bibliotek ręcznie.

## Wymagania wstępne

Na gołym Windowsie wystarczy **jedno** z dwóch — nie potrzeba `make`, `git-bash` ani innych narzędzi:

| Ścieżka | Jedyne wymaganie | Co dostajesz |
|---|---|---|
| **A — Docker** | Docker Desktop / Docker Compose | UI + API + MLflow jednym poleceniem, bez Pythona |
| **B — Python** | Python 3.11–3.13 | uruchomienie przez dołączone skrypty `setup.ps1` / `run.ps1` |

---

## Jak czytać ten system

Najprostszy obraz projektu jest taki: to narzędzie do operacyjnej wyceny laptopów, a nie sam
model regresyjny. Osoba biznesowa może użyć UI do sprawdzenia pojedynczej oferty albo całej
partii sprzętu z CSV. Osoba techniczna może użyć API, żeby podłączyć tę samą predykcję do
innego systemu. Osoba odpowiedzialna za model widzi metryki, drift danych, retraining i runy
MLflow, czyli elementy potrzebne do utrzymania modelu po pierwszym wdrożeniu.

W praktyce przepływ wygląda tak: specyfikacja laptopa trafia do UI albo API, payload jest
walidowany, model zwraca cenę, a system pokazuje przedział typowego błędu i wpływ cech. Gdy
pojawią się nowe dane, można najpierw sprawdzić ich schemat i drift, a dopiero potem uruchomić
retraining. Nowy artefakt jest promowany tylko wtedy, gdy przejdzie bramki jakości.

---

## Wariant A — Docker (najprostszy, zalecany)

**Czego potrzebujesz:** tylko [Docker Desktop](https://www.docker.com/products/docker-desktop/)
(Windows / macOS / Linux). Nic więcej — Python i wszystkie biblioteki instalują się same
wewnątrz kontenera.

```bash
git clone https://github.com/nimzoi/SUML-project.git
cd SUML-project
docker compose up --build
```

Po zbudowaniu (model trenuje się automatycznie podczas budowy obrazu, ~1 min) otwórz:

- **Aplikacja (UI):** http://localhost:8501
- **API + interaktywna dokumentacja:** http://localhost:8000/docs
- **MLflow UI (historia treningów):** http://localhost:5000

Aby zatrzymać: `Ctrl+C`, a następnie `docker compose down`.

> **Na prezentację (sala):** zbuduj obraz wcześniej (`docker compose build`) — na miejscu
> wystarczy wtedy `docker compose up` (sekundy zamiast minut, bez zależności od internetu na sali).

---

## Co działa na którym porcie

| Usługa | Adres | Rola w procesie |
|---|---|---|
| Streamlit UI | http://localhost:8501 | Panel operacyjny dla użytkownika: wycena, CSV, eksport, wykresy, monitoring |
| FastAPI | http://localhost:8000 | Warstwa integracyjna dla innych aplikacji i procesów batch |
| Swagger UI | http://localhost:8000/docs | Testowanie API i walidacji payloadów bez pisania klienta |
| ReDoc | http://localhost:8000/redoc | Czytelna specyfikacja kontraktów API |
| MLflow UI | http://localhost:5000 | Historia eksperymentów, metryk i artefaktów modelu (w Dockerze startuje automatycznie) |

Docker Compose startuje UI, API **oraz** MLflow UI (usługa `mlflow`). W wariancie lokalnym
(bez Dockera) MLflow jest opcjonalny — panel uruchomisz osobno poleceniem
`MLFLOW_ALLOW_FILE_STORE=true python -m mlflow ui --backend-store-uri mlruns --host 127.0.0.1 --port 5000`.

---

## Wariant B — lokalnie, bez Dockera (Python)

**Czego potrzebujesz:** Python 3.11–3.13 (3.14 jeszcze nieobsługiwany) ([python.org](https://www.python.org/downloads/)).

**Windows — najprościej (dwa polecenia, bez `make`):**

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1   # instalacja zaleznosci (raz, tworzy .venv)
powershell -ExecutionPolicy Bypass -File .\run.ps1     # uruchomienie aplikacji (UI)
```

Albo ręcznie (każdy system):

```bash
git clone https://github.com/nimzoi/SUML-project.git
cd SUML-project
python -m pip install -r requirements-dev.txt   # instalacja zależności
python -m model.train                            # opcjonalnie: wytrenowany model jest już w repo
streamlit run app/ui.py                           # uruchom aplikację (UI)
```

Aplikacja otworzy się w przeglądarce pod http://localhost:8501.

Opcjonalnie, aby uruchomić **API** w osobnym terminalu:

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

> **Windows:** powyższe komendy działają w PowerShell. Skróty `make api` / `make ui`
> wymagają narzędzia `make` (na Windows zwykle go nie ma) — używaj pełnych komend powyżej.
>
> **Linux:** jeśli pojawi się błąd LightGBM o brakującej bibliotece `libgomp`, zainstaluj ją:
> `sudo apt-get install libgomp1`.

---

## Wariant C — Streamlit Community Cloud (online, bez instalacji)

Aplikacja działa też samodzielnie w chmurze (UI ładuje model bezpośrednio, gdy nie ma API):

1. Upewnij się, że repozytorium jest publiczne (lub autoryzuj Streamlit dla prywatnego).
2. Na [share.streamlit.io](https://share.streamlit.io): **New app** → repo `nimzoi/SUML-project`,
   gałąź `main`, plik główny `app/ui.py` → **Deploy**.
3. Zależności pobierają się z `requirements.txt`, a `packages.txt` instaluje `libgomp1`.
   Model trenuje się raz przy pierwszym uruchomieniu (~60 s), potem jest zapamiętany.

---

## Jak używać aplikacji

1. Wybierz **gotowy preset** (np. „Gamingowy") albo wpisz własną specyfikację.
2. Kliknij **„Oszacuj cenę"**.
3. Zobaczysz: szacowaną **cenę w PLN z przedziałem**, panel **„Dlaczego ta cena?"**
   (wpływ poszczególnych cech), wykres **„Co jeśli więcej RAM?"** oraz panel
   **„Monitoring danych"** z raportem driftu aktualnego datasetu.
4. W panelu **„Wycena wsadowa CSV"** możesz pobrać przykładowy plik, wgrać listę laptopów,
   policzyć ceny hurtowo i pobrać wynikowy CSV z ceną oraz przedziałem błędu.

Wycena przez API (`POST /predict`):

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{
  \"company\": \"Dell\", \"type_name\": \"Notebook\", \"inches\": 15.6, \"ram_gb\": 8,
  \"weight_kg\": 1.6, \"touchscreen\": 0, \"ips\": 1, \"ppi\": 141.2,
  \"cpu_brand\": \"Intel Core i5\", \"ssd_gb\": 256, \"hdd_gb\": 0,
  \"gpu_brand\": \"Intel\", \"os\": \"Windows\" }"
```

API obsługuje też predykcje wsadowe przez `POST /predict-batch`, wyjaśnienia przez
`POST /explain` oraz endpointy operacyjne `GET /data-schema` i `GET /data-drift`.
Pierwszy pokazuje wymagane kolumny CSV, cechy modelowe i bramki walidacji, a drugi
porównuje aktualny dataset z profilem zapisanym przy treningu.

Pełna mapa endpointów:

| Endpoint | Po co istnieje w procesie |
|---|---|
| `GET /health` | monitoring, czy usługa predykcyjna jest gotowa do pracy |
| `POST /predict` | szybka wycena jednej oferty |
| `POST /explain` | uzasadnienie wyceny i pokazanie wpływu cech |
| `POST /predict-batch` | automatyczna wycena większej partii laptopów |
| `GET /model-info` | audyt aktualnego modelu: metryki, data treningu, MLflow run |
| `GET /data-schema` | kontrakt dla osoby przygotowującej nowy plik danych |
| `GET /data-drift` | kontrola, czy nowe dane są podobne do danych użytych w treningu |
| `POST /validate-data` | sprawdzenie datasetu przed retrainingiem |
| `POST /retrain` | bezpieczne uruchomienie nowego treningu w tle |
| `GET /retrain/{job_id}` | śledzenie, czy retraining przeszedł walidację i promocję artefaktu |

---

## Trening na nowych danych (retrening)

Model można dotrenować na świeższych danych **bez zmian w kodzie**:

1. Podmień plik `data/raw/laptop_data.csv` na nowy — **z tymi samymi kolumnami**
   (`Company`, `TypeName`, `Inches`, `ScreenResolution`, `Cpu`, `Ram`, `Memory`, `Gpu`,
   `OpSys`, `Weight`, `Price`).
2. Uruchom `python -m model.train`.
3. Pliki `model/artifacts/model.joblib` i `metrics.json` zostaną przebudowane; aplikacja
   użyje nowego modelu po restarcie.

> **Uwaga:** warstwa czyszczenia danych (`data/features.py`) jest dopasowana do powyższego
> schematu kolumn. Zbiór o innym układzie kolumn wymaga dostosowania `data/features.py`.

Parametry treningu (budżet czasu AutoML, lista estymatorów, metryka itd.) zmienia się
w pliku `config.yaml` — to jedyne źródło konfiguracji.

Jeśli `tracking.mlflow.enabled` jest ustawione na `true`, trening zapisuje run w lokalnym
katalogu `mlruns/`. Panel eksperymentów uruchomisz komendą:

```bash
MLFLOW_ALLOW_FILE_STORE=true mlflow ui --backend-store-uri mlruns --host 127.0.0.1 --port 5000
```

W projekcie z `make` można użyć skrótu `make mlflow` (ustawia tę zmienną automatycznie).
MLflow 3.x wymaga `MLFLOW_ALLOW_FILE_STORE=true` dla lokalnego file store — `make mlflow`
oraz Docker robią to za Ciebie.

---

## Retrening przez API i walidacja

FastAPI ma interaktywną dokumentację Swagger pod http://localhost:8000/docs. Operacyjny
pipeline retreningu jest dostępny z API:

```bash
curl -X POST http://localhost:8000/validate-data
curl -X GET http://localhost:8000/data-drift
curl -X POST http://localhost:8000/retrain -H "Content-Type: application/json" -d "{}"
```

`POST /validate-data` sprawdza aktualny dataset bez trenowania modelu. Dane tabelaryczne
są walidowane przez Pandera, a konfiguracja i payloady API przez Pydantic. `GET /data-drift`
porównuje aktualny dataset z profilem zapisanym w `metrics.json`; jeśli podmienione dane
mają wyraźnie inne średnie lub rozkłady kategorii, raport wskaże cechy przekraczające progi.
`POST /retrain` uruchamia job w tle: dane są walidowane, model trenuje się w katalogu
tymczasowym, a artefakty są podmieniane dopiero po przejściu bramek jakości z sekcji
`validation` w `config.yaml`. Status sprawdzisz przez `GET /retrain/{job_id}`. Jeśli
aplikacja działa publicznie, ustaw `RETRAIN_API_KEY`; wtedy wywołanie retreningu wymaga
nagłówka `X-API-Key`.

---

## Rozwiązywanie problemów

| Problem | Rozwiązanie |
|---|---|
| Port 8501/8000 zajęty | Zatrzymaj inny proces lub zmień port (`streamlit run app/ui.py --server.port 8600`) |
| `docker compose` nie działa | Upewnij się, że Docker Desktop jest uruchomiony |
| Błąd LightGBM o `libgomp` (Linux) | `sudo apt-get install libgomp1` |
| `make: command not found` (Windows) | Użyj pełnych komend zamiast skrótów `make` (patrz Wariant B) |
| UI pokazuje „Model niedostępny" | Uruchom najpierw `python -m model.train` |

---

## Weryfikacja jakości

```bash
make test     # testy (pytest)         — lub: pytest
make lint     # pylint (>= 8; obecnie 10.00/10)
make format   # formatowanie (black + isort)
make validate # walidacja konfiguracji i danych
make mlflow   # lokalny panel MLflow
```

CI/CD (GitHub Actions) waliduje konfigurację, uruchamia `pylint` i `pytest` przy każdym
pushu i pull requeście. Po pushu do `main` buduje obraz Dockera i publikuje go do GitHub
Container Registry.
Pełny opis architektury i danych: [README.md](../README.md) oraz [data_card.md](data_card.md).
