"""Streamlit entry point, kept at the repository root.

Running `streamlit run app/ui.py` would put only `app/` on the import path, which
breaks `import app...` (ModuleNotFoundError: No module named 'app'). Keeping the
entry point at the root makes the `app`, `config`, `data` and `model` packages
importable both locally and on Streamlit Community Cloud.
"""

from app.ui import main

if __name__ == "__main__":
    main()
