# Data Dictionary — CarDekho Used-Car Dataset (v1)

**File:** `data/cardekho_used_cars.csv` (4,340 rows incl. ~763 exact duplicates removed during cleaning → ~3,577 usable)
**Canonical source:** https://www.kaggle.com/datasets/manishkr1754/cardekho-used-car-data (15,411-row extended variant)
**Mirror used (no Kaggle auth required):** https://github.com/IMvision12/Car-Price_prediction/blob/main/CAR%20DETAILS%20FROM%20CAR%20DEKHO.csv
**License / use:** public educational use; CarDekho listings. Prices reflect listing period (up to 2020), not current market.

| Column | Meaning | Example |
|---|---|---|
| `name` | Free-text brand + model + variant | "Hyundai Creta 1.6 SX" |
| `year` | Manufacturing year (1992–2020) | 2015 |
| `selling_price` | **Target.** Historical asking price in INR | 600000 |
| `km_driven` | Kilometres driven | 70000 |
| `fuel` | Petrol / Diesel / CNG / LPG / Electric (1) | Diesel |
| `seller_type` | Individual / Dealer / Trustmark Dealer | Individual |
| `transmission` | Manual / Automatic | Manual |
| `owner` | First / Second / Third / Fourth & Above / Test Drive Car | First Owner |

**Derived features (training only):**
- `brand` = first token of `name` (reduces 1,491 variants → ~30 brands)
- `car_age` = reference_year − `year` (reference = current year at train/predict time)

**Leakage note:** this dataset has no `present_price` / ex-showroom column, so no
target-leaking feature exists. If the richer 15k-row variant (with `mileage`,
`engine`, `max_power`, `seats`) is adopted later, the pipeline auto-detects extra
columns; any feature revealing the target must be excluded per PRD §6.
