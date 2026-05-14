# Eksperimen_SML_Athalie-Aurora

## California Housing Dataset - Preprocessing Pipeline

**Nama:** Athalie Aurora  
**Dataset:** California Housing (scikit-learn built-in)  
**Task:** Regression - Prediksi Median House Value

---

## Struktur Repository

```
Eksperimen_SML_Athalie-Aurora/
├── .github/
│   └── workflows/
│       └── preprocessing.yml       ← GitHub Actions (Advance)
├── california_housing_raw/
│   └── california_housing_raw.csv  ← Raw dataset
├── preprocessing/
│   ├── Eksperimen_Athalie-Aurora.ipynb     ← Notebook eksperimen
│   ├── automate_Athalie-Aurora.py          ← Script otomatisasi (Skilled)
│   └── california_housing_preprocessing/
│       ├── train.csv               ← Data train siap latih
│       └── test.csv                ← Data test siap latih
└── README.md
```

---

## Cara Menjalankan

### Manual (Notebook)

Buka dan jalankan `preprocessing/Eksperimen_Athalie-Aurora.ipynb`

### Otomatis (Script)

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
cd preprocessing
python automate_Athalie-Aurora.py
```

### GitHub Actions

Workflow otomatis terpantik saat push ke branch `main` pada folder `preprocessing/` atau `california_housing_raw/`.

---

## Tahapan Preprocessing

1. **Data Loading** — Load dari scikit-learn, simpan raw CSV
2. **EDA** — Distribusi, korelasi, deteksi outlier
3. **Handle Missing Values** — SimpleImputer (median)
4. **Handle Outliers** — IQR Clipping (1.5x)
5. **Feature Engineering** — 3 fitur baru
6. **Train-Test Split** — 80:20, random_state=42
7. **Feature Scaling** — StandardScaler
