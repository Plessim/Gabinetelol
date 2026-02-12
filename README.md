# Gabinete LoL Web (Streamlit)

Este repositório está preparado para deploy no **Streamlit Community Cloud**.

## Rodar localmente (Windows)

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy no Streamlit Community Cloud

1. Suba este projeto para o GitHub (repositório público ou privado).
2. No Streamlit Cloud: **New app → From GitHub**
3. Selecione o repo/branch e informe o **Main file**: `streamlit_app.py`
4. Deploy.

### Observações importantes
- O Cloud roda em Linux; por isso o projeto usa caminhos relativos e evita `C:\...`.
- Arquivos de `secrets` não devem ir para o GitHub. Se precisar, use **App → Settings → Secrets** no Cloud.
- O disco no Cloud pode ser temporário. Se você salvar coisas em JSON durante o uso, isso pode não persistir após reinícios do app.
