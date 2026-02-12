"""Entrada padrão para Streamlit Cloud.

Se você quiser, configure o 'Main file path' do Streamlit Cloud como `streamlit_app.py`.
Este wrapper garante que o app principal (plays_app.py) rode com imports locais resolvidos.
"""

from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import plays_app  # noqa: F401
