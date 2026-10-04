from sklearn.datasets import fetch_openml

data = fetch_openml(
    data_id=41214,
    as_frame=True
)

df = data.frame

# Tirage reproductible de 20 000 contrats
sample = df.sample(
    n=20_000,
    random_state=42
).copy()

# Renommage des variables pour rendre le projet plus lisible
sample = sample.rename(columns={
    "IDpol": "policy_id",
    "ClaimNb": "claim_count",
    "Exposure": "exposure",
    "Area": "area",
    "VehPower": "vehicle_power",
    "VehAge": "vehicle_age",
    "DrivAge": "driver_age",
    "BonusMalus": "bonus_malus",
    "VehBrand": "vehicle_brand",
    "VehGas": "vehicle_fuel",
    "Density": "density",
    "Region": "region"
})

# Sauvegarde du portefeuille de démonstration
sample.to_csv(
    "data/sample_portfolio.csv",
    index=False
)

print(sample.head())
print()
print("Dimensions :", sample.shape)
print("Nombre de sinistres :", sample["claim_count"].sum())
print("Exposition totale :", sample["exposure"].sum())

frequency = (
    sample["claim_count"].sum()
    / sample["exposure"].sum()
)

print(f"Fréquence globale : {frequency:.2%}")