import pandas as pd

def parse_custom_words(uploaded_file):
    if uploaded_file is None:
        return []
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, header=None)
        if df.empty:
            return []
        return df.iloc[:, 0].dropna().astype(str).tolist()
    text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
    return text.replace(",", "\n").splitlines()

def make_download_name(prefix, extension):
    return f"{prefix}.{extension.lstrip('.')}"
