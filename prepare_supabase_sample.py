import pandas as pd
from pathlib import Path

source = Path("data/processed/official/normalized_observations.csv")
output = Path("data/processed/official/supabase_sample.csv")

df = pd.read_csv(source)

# Keep valid values within preliminary screening range
df = df[
    df["source_value"].notna()
    & (df["source_value"].abs() <= 100)
].copy()

# Select maximum 200 records per state and measurement type
parts = []

for _, group in df.groupby(["state", "measurement_type"]):
    parts.append(
        group.sample(
            n=min(len(group), 200),
            random_state=42
        )
    )

sample = pd.concat(parts, ignore_index=True)

sample.to_csv(output, index=False)

print("\n===== SUPABASE SAMPLE =====")
print("Sample records:", len(sample))
print("States:", sample["state"].nunique())
print("Measurement types:", sample["measurement_type"].nunique())
print("File size MB:", round(output.stat().st_size / 1024**2, 2))
print("Saved:", output.resolve())