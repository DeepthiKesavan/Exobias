# Data folder

Place your NASA Kepler KOI CSV here (or upload it directly in the app).

## Recommended dataset: Kepler DR25 KOI table

Direct CSV download (browser): `https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+q1_q17_dr25_koi&format=csv`  
If the saved file has no `.csv` extension, rename it to e.g. `koi_dr25.csv`.

Manual route:

1. Go to <https://exoplanetarchive.ipac.caltech.edu/>
2. Open **Data -> Kepler Objects of Interest -> Cumulative / DR25 Supplemental** table
   (table name: `q1_q17_dr25_koi`, the "Q1-Q17 DR25 KOI" table).
3. Click **Download Table**, choose **CSV**, include all columns (at minimum:
   `kepid, kepoi_name, koi_disposition, koi_score, koi_period, koi_prad, koi_teq,
   koi_insol, koi_steff, koi_srad, koi_smass`).
4. Save the file here, e.g. `data/koi_dr25.csv`.

NASA archive CSVs begin with `#` comment lines; ExoBias skips them automatically.

## Known properties of the DR25 table

* `koi_disposition` is mainly `CANDIDATE` or `FALSE POSITIVE`. `CONFIRMED` may be absent.
  ExoBias runs the main experiments with whichever supported classes are present.
* `koi_score` is the Robovetter disposition score. It is **not** a calibrated probability
  that an object is a real planet.
* Never commit large CSV files; `.gitignore` excludes `data/*.csv`.
