# Instrukcja instalacji i uruchomienia — Wycena laptopa

Aplikacja szacuje cenę laptopa na podstawie jego specyfikacji. Dostępna jest jako
**strona (Streamlit)** oraz **API (FastAPI)**. Ten dokument prowadzi krok po kroku — od
zera do działającej aplikacji — bez konieczności konfigurowania czegokolwiek w systemie.

> Wskazówka: jeśli chcesz tylko zobaczyć, jak działa — wybierz **Wariant A (Docker)**.
> To jedna komenda i nie wymaga instalowania Pythona ani żadnych bibliotek ręcznie.

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

Aby zatrzymać: `Ctrl+C`, a następnie `docker compose down`.

---

## Wariant B — lokalnie, bez Dockera (Python)

**Czego potrzebujesz:** Python 3.11 lub nowszy ([python.org](https://www.python.org/downloads/)).

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
   (wpływ poszczególnych cech) oraz wykres **„Co jeśli więcej RAM?"**.

Wycena przez API (`POST /predict`):

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{
  \"company\": \"Dell\", \"type_name\": \"Notebook\", \"inches\": 15.6, \"ram_gb\": 8,
  \"weight_kg\": 1.6, \"touchscreen\": 0, \"ips\": 1, \"ppi\": 141.2,
  \"cpu_brand\": \"Intel Core i5\", \"ssd_gb\": 256, \"hdd_gb\": 0,
  \"gpu_brand\": \"Intel\", \"os\": \"Windows\" }"
```

---

## Trening na nowych danych (retrening)

Model można dotrenować na świeższych danych **bez zmian w kodzie**:

1. Podmień plik `data/raw/laptop_data.csv` na nowy — **z tymi samymi kolumnami**
   (`Company`, `TypeName`, `Inches`, `ScreenResolution`, `Cpu`, `Ram`, `Memory`, `Gpu`,
   `OpSys`, `Weight`, `Price`).
2. Uruchom `python -m model.train`.
3. Pliki `model/artifacts/model.joblib` i `metrics.json` zostaną przebudowane; aplikacja
   użyje nowego modelu po restarcie.
4. **Zacommituj** przebudowane pliki (są śledzone w repo), żeby cała grupa miała aktualny
   model bez ponownego trenowania.

> **Uwaga:** warstwa czyszczenia danych (`data/features.py`) jest dopasowana do powyższego
> schematu kolumn. Zbiór o innym układzie kolumn wymaga dostosowania `data/features.py`.

Parametry treningu (budżet czasu AutoML, lista estymatorów, metryka itd.) zmienia się
w pliku `config.yaml` — to jedyne źródło konfiguracji.

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

## Weryfikacja jakości (dla oceniającego)

```bash
make test     # testy (pytest)         — lub: pytest
make lint     # pylint (>= 8; obecnie 10.00/10)
make format   # formatowanie (black + isort)
```

CI (GitHub Actions) uruchamia `pylint` i `pytest` przy każdym push i pull requeście.
Pełny opis architektury i danych: [README.md](../README.md) oraz [data_card.md](data_card.md).
