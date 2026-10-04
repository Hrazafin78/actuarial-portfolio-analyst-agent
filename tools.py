import pandas as pd
import plotly.express as px


def load_and_validate_data(filepath):
    """
    Charge un portefeuille automobile depuis un fichier CSV
    et vérifie que les données essentielles sont valides.
    """

    df = pd.read_csv(filepath)

    required_columns = [
        "policy_id",
        "exposure",
        "claim_count"
    ]

    # Vérification des colonnes obligatoires
    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Colonnes manquantes : {missing_columns}"
        )

    # L'exposition doit être strictement positive
    if (df["exposure"] <= 0).any():
        raise ValueError(
            "L'exposition doit être strictement positive."
        )

    # Le nombre de sinistres ne peut pas être négatif
    if (df["claim_count"] < 0).any():
        raise ValueError(
            "Le nombre de sinistres ne peut pas être négatif."
        )

    return df

def portfolio_summary(df):
    """
    Retourne les principaux indicateurs du portefeuille.
    """

    total_policies = len(df)
    total_exposure = df["exposure"].sum()
    total_claims = df["claim_count"].sum()

    frequency = total_claims / total_exposure

    return {
        "number_of_policies": total_policies,
        "total_exposure": total_exposure,
        "total_claims": total_claims,
        "claim_frequency": frequency
    }

def claim_frequency(df):
    """
    Calcule la fréquence de sinistres du portefeuille.
    """

    total_claims = df["claim_count"].sum()
    total_exposure = df["exposure"].sum()

    frequency = total_claims / total_exposure

    return {
        "total_claims": total_claims,
        "total_exposure": total_exposure,
        "claim_frequency": frequency
    }

def frequency_by_segment(df, variable, min_exposure=0):
    """
    Calcule la fréquence de sinistres par modalité
    d'une variable du portefeuille.
    """

    if variable not in df.columns:
        raise ValueError(
            f"La variable '{variable}' n'existe pas dans le portefeuille."
        )

    result = (
        df.groupby(variable, observed=True)
        .agg(
            exposure=("exposure", "sum"),
            claims=("claim_count", "sum"),
            policies=("policy_id", "count")
        )
        .reset_index()
    )

    result["frequency"] = (
        result["claims"] / result["exposure"]
    )

    # Garde uniquement les segments suffisamment exposés
    result = result[
        result["exposure"] >= min_exposure
    ]

    return result


df = load_and_validate_data("data/sample_portfolio.csv")

result = frequency_by_segment(
    df,
    variable="vehicle_fuel",
    min_exposure=100
)


def risk_segments(df, variable, top_n=5, min_exposure=100):
    """
    Identifie les segments ayant les fréquences de sinistres
    les plus élevées, sous réserve d'une exposition minimale.
    """

    result = frequency_by_segment(
        df,
        variable=variable,
        min_exposure=min_exposure
    )

    result = result.sort_values(
        by="frequency",
        ascending=False
    )

    return result.head(top_n)


def detect_anomalies(df):
    """
    Détecte les principales anomalies de qualité
    dans le portefeuille.
    """

    anomalies = {}

    # Valeurs manquantes
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]

    anomalies["missing_values"] = missing_values.to_dict()

    # Expositions nulles ou négatives
    anomalies["invalid_exposure"] = int(
        (df["exposure"] <= 0).sum()
    )

    # Nombre de sinistres négatif
    anomalies["negative_claim_count"] = int(
        (df["claim_count"] < 0).sum()
    )

    # Doublons d'identifiants
    anomalies["duplicate_policy_id"] = int(
        df["policy_id"].duplicated().sum()
    )

    return anomalies

def filter_portfolio(df, variable, operator, value):
    """
    Filtre le portefeuille selon une condition.

    Opérateurs disponibles :
    ==, !=, <, <=, >, >=
    """

    if variable not in df.columns:
        raise ValueError(
            f"La variable '{variable}' n'existe pas dans le portefeuille."
        )

    operators = {
        "==": lambda x: x == value,
        "!=": lambda x: x != value,
        "<": lambda x: x < value,
        "<=": lambda x: x <= value,
        ">": lambda x: x > value,
        ">=": lambda x: x >= value
    }

    if operator not in operators:
        raise ValueError(
            f"Opérateur non reconnu : {operator}"
        )

    mask = operators[operator](df[variable])

    return df[mask].copy()
    
def plot_frequency(df, variable, min_exposure=0):
    """
    Crée un graphique de la fréquence de sinistres
    par segment.
    """

    result = frequency_by_segment(
        df,
        variable=variable,
        min_exposure=min_exposure
    )

    result = result.sort_values(
        by="frequency",
        ascending=False
    )

    fig = px.bar(
        result,
        x=variable,
        y="frequency",
        hover_data=["exposure", "claims", "policies"],
        title=f"Fréquence de sinistres par {variable}",
        labels={
            variable: variable,
            "frequency": "Fréquence de sinistres"
        }
    )

    fig.update_yaxes(
        tickformat=".1%"
    )

    return fig

def filter_portfolio(df, variable, operator, value):
    """
    Filtre le portefeuille selon une condition.

    Opérateurs disponibles :
    ==, !=, <, <=, >, >=
    """

    if variable not in df.columns:
        raise ValueError(
            f"La variable '{variable}' n'existe pas dans le portefeuille."
        )

    operators = {
        "==": lambda x: x == value,
        "!=": lambda x: x != value,
        "<": lambda x: x < value,
        "<=": lambda x: x <= value,
        ">": lambda x: x > value,
        ">=": lambda x: x >= value
    }

    if operator not in operators:
        raise ValueError(
            f"Opérateur non reconnu : {operator}"
        )

    mask = operators[operator](df[variable])

    return df[mask].copy()


def filtered_frequency(df, variable, operator, value):
    """
    Calcule la fréquence de sinistres sur une sous-population
    définie par un filtre.
    """

    filtered_df = filter_portfolio(
        df,
        variable=variable,
        operator=operator,
        value=value
    )

    if filtered_df.empty:
        raise ValueError(
            "Le filtre ne retourne aucun contrat."
        )

    result = claim_frequency(filtered_df)

    result["number_of_policies"] = len(filtered_df)

    return result