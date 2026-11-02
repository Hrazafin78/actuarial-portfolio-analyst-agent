"""Moteur statistique et actuariel des portefeuilles IARD.

Valide le mapping manuel, calcule statistiques, ratios, associations et
agrégats par segment ou année. Construit un contexte IA borné, sans appeler
de modèle ni modifier le portefeuille fourni.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


ROLES = ("numeric", "categorical", "temporal", "identifier", "exclude")
ACTUARIAL_FIELDS = ("claim_count", "exposure", "premium", "claim_amount", "year", "policy_id")
MEASURES = ("claim_count", "exposure", "premium", "claim_amount")


@dataclass(frozen=True)
class AnalysisOptions:
    """Paramètres immuables des bornes de distribution, segmentation et associations."""
    top_categories: int = 20
    max_categories_association: int = 80
    max_categories_segment: int = 80
    numerical_bins: int = 8
    min_pairwise_observations: int = 3

    def __post_init__(self) -> None:
        """Refuse les bornes inférieures à deux à la construction des options."""
        if min(self.top_categories, self.max_categories_association,
               self.max_categories_segment, self.numerical_bins,
               self.min_pairwise_observations) < 2:
            raise ValueError("Toutes les limites d'analyse doivent être >= 2")


def _check_df(df: pd.DataFrame) -> None:
    """Exige un DataFrame dont les noms de colonnes sont des chaînes uniques."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df doit être un DataFrame pandas")
    if not df.columns.is_unique:
        raise ValueError("Les noms de colonnes doivent être uniques")
    if any(not isinstance(c, str) for c in df.columns):
        raise ValueError("Les noms de colonnes doivent être des chaînes")


def _numeric(series: pd.Series) -> pd.Series:
    """Conversion sûre, conservant NaN et éliminant les infinis."""
    num = pd.to_numeric(series, errors="coerce")
    num = num.astype("float64")
    return num.replace([np.inf, -np.inf], np.nan)


def detect_variable_roles(df: pd.DataFrame, overrides: Mapping[str, str] | None = None) -> dict[str, str]:
    """Détermine la nature statistique, PAS le rôle actuariel, des colonnes.

    Les rôles sont toujours modifiables par l'utilisateur. Un code numérique
    et un identifiant mal nommé doivent notamment être corrigés dans l'UI.
    """
    _check_df(df)
    result: dict[str, str] = {}
    for col in df.columns:
        series = df[col]
        name = col.casefold().strip()
        if pd.api.types.is_datetime64_any_dtype(series):
            role = "temporal"
        elif name in {"year", "annee", "année", "underwriting_year", "calendar_year"}:
            role = "temporal"
        elif (name == "id" or name.endswith("_id") or name.startswith("id_")
              or name in {"idpol", "policy_number", "contract_number"}):
            role = "identifier"
        elif pd.api.types.is_bool_dtype(series):
            role = "categorical"
        elif pd.api.types.is_numeric_dtype(series):
            role = "numeric"
        else:
            role = "categorical"
        result[col] = role
    for col, role in (overrides or {}).items():
        if col not in result:
            raise ValueError(f"Colonne inconnue pour le rôle : {col}")
        if role not in ROLES:
            raise ValueError(f"Rôle inconnu pour {col} : {role}")
        result[col] = role
    return result


def validate_actuarial_mapping(
    df: pd.DataFrame, mapping: Mapping[str, str | None] | None,
) -> dict[str, str]:
    """Validation de la sélection manuelle (aucune détection heuristique)."""
    _check_df(df)
    mapping = mapping or {}
    if any(k not in ACTUARIAL_FIELDS for k in mapping):
        raise ValueError("Rôle actuariel inconnu")
    result = {key: value for key, value in mapping.items() if value not in (None, "")}
    for key, col in result.items():
        if not isinstance(col, str) or col not in df.columns:
            raise ValueError(f"Variable absente pour {key} : {col}")
    if len(set(result.values())) != len(result):
        raise ValueError("Une colonne ne peut pas correspondre à deux rôles actuariels")
    for key in MEASURES:
        if key not in result:
            continue
        col = result[key]
        raw = df[col]
        numeric = _numeric(raw)
        invalid = raw.notna() & numeric.isna()
        if raw.isna().any():
            raise ValueError(f"{col} : valeurs manquantes à traiter avant un calcul actuariel")
        if invalid.any():
            raise ValueError(f"{col} : valeurs non numériques ou infinies pour {key}")
        if key in {"claim_count", "exposure"}:
            if (numeric.dropna() < 0).any():
                raise ValueError(f"{col} : valeurs négatives interdites pour {key}")
        if key == "claim_count":
            non_null = numeric.dropna()
            if ((non_null % 1) != 0).any():
                raise ValueError(f"{col} : nombre de sinistres non entier")
    return result


def analyze_dataset(df: pd.DataFrame, roles: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Décrit la structure, les manquants et les rôles statistiques de chaque colonne."""
    _check_df(df)
    roles = detect_variable_roles(df, roles)
    n = len(df)
    out: dict[str, Any] = {
        "n_rows": n, "n_columns": len(df.columns),
        "total_missing": int(df.isna().sum().sum()),
        "missing_pct": float(df.isna().sum().sum() / (max(n * len(df.columns), 1)) * 100),
        "role_counts": {role: sum(x == role for x in roles.values()) for role in ROLES},
        "columns": pd.DataFrame([
            {"variable": c, "role": roles[c], "dtype": str(df[c].dtype),
             "n_non_null": int(df[c].notna().sum()),
             "n_missing": int(df[c].isna().sum()),
             "n_unique": int(df[c].nunique(dropna=True))}
            for c in df.columns
        ]),
    }
    return out


def analyze_numeric_variables(df: pd.DataFrame, roles: Mapping[str, str]) -> pd.DataFrame:
    """Calcule les statistiques univariées sur les valeurs numériques finies renseignées."""
    rows = []
    for col, role in roles.items():
        if role != "numeric":
            continue
        x = _numeric(df[col]); v = x.dropna()
        q = v.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]) if len(v) else None
        rows.append({
            "variable": col, "count": len(v), "missing": int(len(df) - len(v)),
            "unique": int(v.nunique()),
            "mean": float(v.mean()) if len(v) else np.nan,
            "median": float(v.median()) if len(v) else np.nan,
            "std": float(v.std(ddof=1)) if len(v) > 1 else np.nan,
            "variance": float(v.var(ddof=1)) if len(v) > 1 else np.nan,
            "min": float(v.min()) if len(v) else np.nan,
            "max": float(v.max()) if len(v) else np.nan,
            "q01": float(q.loc[0.01]) if q is not None else np.nan,
            "q05": float(q.loc[0.05]) if q is not None else np.nan,
            "q25": float(q.loc[0.25]) if q is not None else np.nan,
            "q75": float(q.loc[0.75]) if q is not None else np.nan,
            "q95": float(q.loc[0.95]) if q is not None else np.nan,
            "q99": float(q.loc[0.99]) if q is not None else np.nan,
            "skewness": float(v.skew()) if len(v) >= 3 else np.nan,
            "zeros": int((v == 0).sum()),
        })
    return pd.DataFrame(rows)


def analyze_categorical_variables(
    df: pd.DataFrame, roles: Mapping[str, str], top_n: int = 20,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Retourne les résumés catégoriels et les distributions bornées au top demandé.

    Regroupe le reste dans une modalité dédiée. Pour les identifiants, ne
    conserve que le résumé, sans distribution des valeurs individuelles.
    """
    if top_n < 2:
        raise ValueError("top_n >= 2 requis")
    summaries, distributions = [], {}
    for col, role in roles.items():
        if role not in ("categorical", "identifier"):
            continue
        values = df[col]
        # Les identifiants peuvent avoir des centaines de milliers de valeurs :
        # éviter de matérialiser leur distribution intégrale.
        counts = values.value_counts(dropna=False) if role != "identifier" else None
        modal_count = int(counts.iloc[0]) if counts is not None and len(counts) else 0
        summaries.append({
            "variable": col, "role": role, "n_unique": int(values.nunique(dropna=True)),
            "missing": int(values.isna().sum()),
            "dominant_pct": (100 * modal_count / len(df) if len(df) else np.nan)
                            if role != "identifier" else np.nan,
        })
        if role == "identifier":
            continue  # Ne pas conserver de distribution nominative d'identifiants.
        shown = counts.head(top_n)
        dist = pd.DataFrame({
            "modalite": ["(Manquant)" if pd.isna(v) else str(v) for v in shown.index],
            "effectif": shown.to_numpy(dtype=int),
            "part_pct": 100 * shown.to_numpy(dtype=float) / len(df) if len(df) else np.nan,
        })
        remaining = int(counts.iloc[top_n:].sum())
        if remaining:
            dist.loc[len(dist)] = ["(Autres modalités)", remaining, 100 * remaining / len(df)]
        distributions[col] = dist
    return pd.DataFrame(summaries), distributions


def _finite_measure(df: pd.DataFrame, mapping: Mapping[str, str], key: str) -> pd.Series | None:
    """Extrait une mesure du mapping, neutralise les non-finis et remplit les manquants à zéro.

    Le parcours de génération valide les mesures avant cette extraction ;
    renvoie None si le rôle n’est pas renseigné.
    """
    if key not in mapping:
        return None
    return _numeric(df[mapping[key]]).fillna(0.0)


def _safe_ratio(num: float | None, den: float | None) -> float | None:
    """Renvoie le quotient, ou None si une mesure manque ou le dénominateur est non positif."""
    if num is None or den is None or den <= 0:
        return None
    return float(num / den)


def _metric_values(sums: Mapping[str, float], n: int) -> dict[str, float | None]:
    """Construit les totaux et ratios globaux à partir des sommes disponibles."""
    claims, exp, premium, cost = (sums.get(key) for key in MEASURES)
    output: dict[str, float | None] = {"observations": n}
    for key in MEASURES:
        if key in sums:
            output[f"total_{key}"] = float(sums[key])
    if claims is not None and exp is not None:
        output["claim_frequency"] = _safe_ratio(claims, exp)
    if cost is not None and claims is not None:
        output["average_severity"] = _safe_ratio(cost, claims)
    if cost is not None and exp is not None:
        output["pure_premium"] = _safe_ratio(cost, exp)
    if cost is not None and premium is not None:
        output["loss_ratio"] = _safe_ratio(cost, premium)
    if premium is not None:
        output["average_premium_per_row"] = _safe_ratio(premium, float(n))
    if premium is not None and exp is not None:
        output["premium_per_exposure"] = _safe_ratio(premium, exp)
    return output


def _actuarial_totals(df: pd.DataFrame, mapping: Mapping[str, str]) -> dict[str, float]:
    """Somme les seules mesures présentes dans le mapping actuariel."""
    return {key: float(_finite_measure(df, mapping, key).sum())
            for key in MEASURES if key in mapping}


def _grouping(df: pd.DataFrame, col: str, role: str, bins: int) -> tuple[pd.Series, str]:
    """Construit les groupes par modalités, valeurs numériques ou tranches quantiles."""
    if role == "numeric":
        x = _numeric(df[col])
        valid = x.dropna()
        if valid.nunique() > bins:
            try:
                return pd.qcut(x, q=bins, duplicates="drop"), "quantiles"
            except ValueError:
                pass
        return x, "valeurs"
    return df[col], "modalites"


def _summarize_groups(
    df: pd.DataFrame,
    groups: pd.Series,
    mapping: Mapping[str, str],
    variable: str,
    method: str,
) -> pd.DataFrame:
    # Groupe '(Manquant)' explicite, sans conversion des données originales.
    """Agrège effectifs, mesures et ratios par groupe, avec une modalité pour les manquants.

    Les parts sont calculées sur le portefeuille fourni ; les dénominateurs
    non positifs donnent des ratios manquants, sans modifier les données sources.
    """
    frame = pd.DataFrame(index=df.index)
    frame["__segment"] = groups.astype("object").map(
        lambda x: "(Manquant)" if pd.isna(x) else str(x)
    )
    frame["__records"] = 1
    for key in MEASURES:
        if key in mapping:
            frame[key] = _finite_measure(df, mapping, key)
    grouped = frame.groupby("__segment", sort=False, observed=True).agg(
        {"__records": "sum", **{key: "sum" for key in MEASURES if key in mapping}}
    ).reset_index().rename(columns={"__segment": "segment", "__records": "observations"})
    grouped.insert(0, "variable", variable)
    grouped.insert(1, "segmentation", method)
    for key in MEASURES:
        if key in mapping:
            grouped.rename(columns={key: f"total_{key}"}, inplace=True)
    for prefix, numerator, denominator in (
        ("claim_frequency", "total_claim_count", "total_exposure"),
        ("average_severity", "total_claim_amount", "total_claim_count"),
        ("pure_premium", "total_claim_amount", "total_exposure"),
        ("loss_ratio", "total_claim_amount", "total_premium"),
        ("premium_per_exposure", "total_premium", "total_exposure"),
        ("average_premium_per_row", "total_premium", "observations"),
    ):
        if numerator in grouped and denominator in grouped:
            den = grouped[denominator].where(grouped[denominator] > 0)
            grouped[prefix] = grouped[numerator].div(den)
    grouped["share_observations_pct"] = grouped["observations"] / len(df) * 100 if len(df) else np.nan
    for key in MEASURES:
        total = f"total_{key}"
        if total in grouped:
            den = grouped[total].sum()
            grouped[f"share_{key}_pct"] = grouped[total] / den * 100 if den > 0 else np.nan
    return grouped


def analyze_portfolio_segments(
    df: pd.DataFrame,
    roles: Mapping[str, str],
    actuarial_mapping: Mapping[str, str],
    *, bins: int = 8,
    max_categories: int = 80,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """KPI par *toute* variable explicative éligible ; pas d'échantillon."""
    result, warnings = {}, []
    excluded = set(actuarial_mapping.values())
    for col, role in roles.items():
        if col in excluded or role not in {"numeric", "categorical"}:
            continue
        x = df[col]
        if role == "categorical" and x.nunique(dropna=True) > max_categories:
            warnings.append(f"Segmentation {col} ignorée : plus de {max_categories} modalités")
            continue
        grouping, method = _grouping(df, col, role, bins)
        if not len(df):
            result[col] = pd.DataFrame()
        else:
            result[col] = _summarize_groups(df, grouping, actuarial_mapping, col, method)
    return result, warnings


def _time_years(series: pd.Series) -> pd.Series:
    """Extrait des années à partir de dates ou de valeurs annuelles reconnues."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.dt.year.astype("Int64")
    x = _numeric(series)
    valid = x.dropna()
    if len(valid) and valid.between(1900, 2200).all() and (valid % 1 == 0).all():
        return x.astype("Int64")
    # Ne pas prendre un numéro de période arbitraire pour une année.
    dates = pd.to_datetime(series, errors="coerce", format="mixed")
    if len(series) and dates.notna().sum() == 0:
        return pd.Series(pd.NA, index=series.index, dtype="Int64")
    return dates.dt.year.astype("Int64")


def analyze_temporal_variables(
    df: pd.DataFrame, roles: Mapping[str, str], actuarial_mapping: Mapping[str, str],
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Retourne les agrégats annuels et avertissements des variables temporelles disponibles."""
    temporal = [c for c, role in roles.items() if role == "temporal"]
    if "year" in actuarial_mapping and actuarial_mapping["year"] not in temporal:
        temporal.append(actuarial_mapping["year"])
    output, warnings = {}, []
    for col in temporal:
        years = _time_years(df[col])
        if df[col].notna().any() and years.notna().sum() == 0:
            warnings.append(f"Temporalité {col} : aucun millésime interprétable")
            continue
        group = _summarize_groups(df, years, actuarial_mapping, col, "annuelle")
        if group.empty:
            output[col] = group
            continue
        # Placer les années non interprétables après les millésimes renseignés.
        order = pd.to_numeric(group["segment"], errors="coerce")
        output[col] = group.assign(_year_order=order).sort_values(
            "_year_order", na_position="last"
        ).drop(columns="_year_order").reset_index(drop=True)
    return output, warnings


def _cramers_v_codes(x: np.ndarray, nx: int, y: np.ndarray, ny: int,
                      min_n: int) -> float:
    """Calcule le V de Cramér corrigé du biais sur des codes catégoriels.

    Écarte les codes manquants ; renvoie NaN si l’effectif ou les dimensions
    de la table de contingence ne permettent pas de définir la mesure.
    """
    valid = (x >= 0) & (y >= 0)
    n = int(valid.sum())
    if n < min_n or nx < 2 or ny < 2:
        return np.nan
    counts = np.bincount(x[valid].astype(np.int64) * ny + y[valid],
                         minlength=nx * ny).reshape(nx, ny)
    counts = counts[np.any(counts, axis=1)][:, np.any(counts, axis=0)]
    r, k = counts.shape
    if min(r, k) < 2:
        return np.nan
    chi2 = chi2_contingency(counts, correction=False)[0]
    phi2_corr = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    return float(np.sqrt(phi2_corr / denom)) if denom > 0 else np.nan


def _eta_sq_codes(codes: np.ndarray, ncat: int, values: np.ndarray, min_n: int) -> float:
    """Calcule la part de variance intergroupes dans la variance numérique totale.

    Écarte les codes manquants et valeurs non finies ; renvoie NaN si les
    conditions d’effectif, de cardinalité ou de variance ne sont pas réunies.
    """
    valid = (codes >= 0) & np.isfinite(values)
    if int(valid.sum()) < min_n or ncat < 2:
        return np.nan
    c = codes[valid]; v = values[valid]
    sizes = np.bincount(c, minlength=ncat)
    sums = np.bincount(c, weights=v, minlength=ncat)
    grand_mean = float(v.mean())
    total = float(np.sum((v - grand_mean) ** 2))
    if total <= 0:
        return np.nan
    avgs = np.divide(sums, sizes, out=np.zeros_like(sums), where=sizes > 0)
    between = float(np.sum(sizes * (avgs - grand_mean) ** 2))
    return float(np.clip(between / total, 0.0, 1.0))


def analyze_relationships(
    df: pd.DataFrame, roles: Mapping[str, str], *, exclude: set[str] | None = None,
    max_categories: int = 80, min_observations: int = 3,
) -> dict[str, Any]:
    """Matrice de Pearson, V de Cramér corrigé et η² sur toutes les paires éligibles.

    Aucun échantillonnage. Les colonnes constantes et de trop forte cardinalité
    sont écartées des mesures non définies/coûteuses avec motif explicite.
    """
    _check_df(df)
    exclude = exclude or set()
    numeric = [c for c, r in roles.items() if r == "numeric" and c not in exclude]
    cats = [c for c, r in roles.items() if r == "categorical" and c not in exclude]
    skipped: list[str] = []
    good_num = []
    arr: dict[str, np.ndarray] = {}
    for c in numeric:
        x = _numeric(df[c])
        if x.nunique(dropna=True) < 2:
            skipped.append(f"{c} : variable numérique constante/vide")
        else:
            good_num.append(c)
            arr[c] = x.to_numpy(dtype=float, na_value=np.nan)
    numeric_frame = pd.DataFrame({c: arr[c] for c in good_num})
    pearson = numeric_frame.corr(method="pearson", min_periods=min_observations)

    good_cats: list[str] = []
    codes: dict[str, np.ndarray] = {}
    levels: dict[str, int] = {}
    for c in cats:
        n_unique = int(df[c].nunique(dropna=True))
        if n_unique < 2:
            skipped.append(f"{c} : variable catégorielle constante/vide")
        elif n_unique > max_categories:
            skipped.append(f"{c} : {n_unique} modalités, limite {max_categories}")
        else:
            arr_codes, unique = pd.factorize(df[c], sort=False, use_na_sentinel=True)
            good_cats.append(c); codes[c] = arr_codes; levels[c] = len(unique)

    cv = pd.DataFrame(np.eye(len(good_cats)), index=good_cats, columns=good_cats)
    eta = pd.DataFrame(index=good_cats, columns=good_num, dtype=float)
    ranking: list[dict[str, Any]] = []

    for a, b in combinations(good_num, 2):
        val = pearson.loc[a, b]
        if pd.notna(val):
            ranking.append({"variable_1": a, "variable_2": b,
                            "method": "Pearson", "coefficient": float(val),
                            "strength": abs(float(val))})
    for a, b in combinations(good_cats, 2):
        val = _cramers_v_codes(codes[a], levels[a], codes[b], levels[b], min_observations)
        cv.loc[a, b] = val; cv.loc[b, a] = val
        if pd.notna(val):
            ranking.append({"variable_1": a, "variable_2": b,
                            "method": "V de Cramér corrigé", "coefficient": float(val),
                            "strength": float(val)})
    for c in good_cats:
        for n in good_num:
            val = _eta_sq_codes(codes[c], levels[c], arr[n], min_observations)
            eta.loc[c, n] = val
            if pd.notna(val):
                ranking.append({"variable_1": c, "variable_2": n,
                                "method": "Eta²", "coefficient": float(val),
                                "strength": float(val)})
    rank = pd.DataFrame(ranking, columns=["variable_1", "variable_2", "method", "coefficient", "strength"])
    if not rank.empty:
        rank = rank.sort_values("strength", ascending=False, kind="stable").reset_index(drop=True)
    return {"pearson": pearson, "cramers_v": cv, "eta_squared": eta,
            "strongest": rank, "skipped": skipped}


def analyze_variable_pair(
    df: pd.DataFrame,
    left: str,
    right: str,
    roles: Mapping[str, str],
    *,
    max_categories: int = 80,
    scatter_sample: int = 8000,
) -> dict[str, Any]:
    """Exploration ciblée de deux variables explicatives.

    Les tableaux sont calculés sur toutes les observations. Seul l'affichage
    du nuage de points est échantillonné, avec graine fixe et taille indiquée.
    """
    _check_df(df)
    if left == right or left not in df or right not in df:
        raise ValueError("Choisir deux variables distinctes et présentes")
    if scatter_sample < 2 or max_categories < 2:
        raise ValueError("Limites invalides")
    a, b = roles[left], roles[right]
    if a not in {"numeric", "categorical"} or b not in {"numeric", "categorical"}:
        raise ValueError("Les deux variables doivent être numériques ou catégorielles")

    if a == b == "numeric":
        pairs = pd.DataFrame({left: _numeric(df[left]), right: _numeric(df[right])}).dropna()
        coefficient = float(pairs[left].corr(pairs[right])) if len(pairs) >= 3 else np.nan
        sample = pairs.sample(n=scatter_sample, random_state=42) if len(pairs) > scatter_sample else pairs
        return {"type": "numeric_numeric", "correlation": coefficient,
                "observations": len(pairs), "plot_sample_size": len(sample),
                "plot_data": sample.reset_index(drop=True)}

    if a == b == "categorical":
        if max(df[left].nunique(dropna=True), df[right].nunique(dropna=True)) > max_categories:
            raise ValueError(f"Tableau croisé limité à {max_categories} modalités par variable")
        counts = pd.crosstab(df[left].fillna("(Manquant)"), df[right].fillna("(Manquant)"), dropna=False)
        fractions = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0) * 100
        return {"type": "categorical_categorical", "counts": counts,
                "row_pct": fractions, "observations": int(counts.to_numpy().sum())}

    categorical, numerical = (left, right) if a == "categorical" else (right, left)
    if df[categorical].nunique(dropna=True) > max_categories:
        raise ValueError(f"Comparaison limitée à {max_categories} modalités")
    numeric = _numeric(df[numerical])
    values = pd.DataFrame({"modalite": df[categorical].fillna("(Manquant)"),
                           "value": numeric})
    grouped = values.groupby("modalite", observed=True, dropna=False)["value"].agg(
        ["count", "mean", "median", "std", "min", "max"]
    ).reset_index()
    valid = values.dropna(subset=["value"])
    sample = valid.sample(n=scatter_sample, random_state=42) if len(valid) > scatter_sample else valid
    return {"type": "categorical_numeric", "statistics": grouped,
            "plot_data": sample.reset_index(drop=True),
            "observations": int(numeric.notna().sum()), "plot_sample_size": len(sample)}


def generate_descriptive_report(
    df: pd.DataFrame,
    *,
    roles: Mapping[str, str] | None = None,
    actuarial_mapping: Mapping[str, str | None] | None = None,
    options: AnalysisOptions | None = None,
) -> dict[str, Any]:
    """Analyse complète, sans muter la base et sans appeler un modèle IA.

    Retourne un dictionnaire contenant seulement des agrégats, des statistiques
    et des DataFrames de résultats (aucune copie du portefeuille brut).
    """
    _check_df(df)
    opts = options or AnalysisOptions()
    actual_roles = detect_variable_roles(df, roles)
    mapping = validate_actuarial_mapping(df, actuarial_mapping)
    overview = analyze_dataset(df, actual_roles)
    num_stats = analyze_numeric_variables(df, actual_roles)
    cat_stats, distributions = analyze_categorical_variables(df, actual_roles, opts.top_categories)
    excluded = set(mapping.values())
    associations = analyze_relationships(
        df, actual_roles, exclude=excluded,
        max_categories=opts.max_categories_association,
        min_observations=opts.min_pairwise_observations,
    )
    segments, warnings = analyze_portfolio_segments(
        df, actual_roles, mapping, bins=opts.numerical_bins,
        max_categories=opts.max_categories_segment,
    )
    temporal, time_warnings = analyze_temporal_variables(df, actual_roles, mapping)
    totals = _actuarial_totals(df, mapping)
    actuarial_overview = _metric_values(totals, len(df))
    if "policy_id" in mapping:
        actuarial_overview["unique_entity_ids"] = int(df[mapping["policy_id"]].nunique(dropna=True))
    if "claim_count" in mapping and "exposure" not in mapping:
        warnings.append("Fréquence indisponible : exposition non renseignée")
    if "claim_amount" in mapping and "premium" not in mapping:
        warnings.append("S/P indisponible : prime non renseignée")
    if "claim_count" in mapping and "exposure" in mapping:
        n_without_exposure = int(((_numeric(df[mapping["claim_count"]]) > 0) &
                                  (_numeric(df[mapping["exposure"]]) == 0)).sum())
        if n_without_exposure:
            warnings.append(f"{n_without_exposure} ligne(s) ont des sinistres avec une exposition nulle : vérifier la cohérence métier")
    if mapping:
        warnings.append("Vérifier la granularité, les unités, périodes et garanties des variables sélectionnées avant toute interprétation actuarielle")
    return {
        "overview": overview,
        "roles": actual_roles,
        "actuarial_mapping": mapping,
        "numeric_stats": num_stats,
        "categorical_stats": cat_stats,
        "category_distributions": distributions,
        "associations": associations,
        "segments": segments,
        "temporal": temporal,
        "actuarial_overview": actuarial_overview,
        "warnings": warnings + time_warnings + associations["skipped"],
    }


def build_llm_context(report: Mapping[str, Any], max_items: int = 15) -> dict[str, Any]:
    """Prépare un contexte agrégé borné pour le modèle, sans lignes individuelles.

    Limite les résumés et exemples au nombre demandé. Un contexte compact
    ne représente pas une consultation exhaustive du rapport.
    """
    if max_items < 1:
        raise ValueError("max_items doit être positif")
    overview = report["overview"]
    associations = report["associations"]["strongest"]
    top = associations.head(max_items).replace([np.inf, -np.inf], np.nan)

    def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
        """Borne un tableau d’agrégats et remplace les valeurs non finies pour le JSON."""
        clean = frame.head(max_items).replace([np.inf, -np.inf], np.nan).astype(object)
        clean = clean.where(pd.notna(clean), None)
        return [
            {k: (v.item() if isinstance(v, np.generic) else v)
             for k, v in row.items()}
            for row in clean.to_dict(orient="records")
        ]

    return {
        "dataset": {"n_rows": overview["n_rows"], "n_columns": overview["n_columns"],
                    "total_missing": overview["total_missing"],
                    "roles": overview["role_counts"]},
        "actuarial_mapping": dict(report["actuarial_mapping"]),
        "actuarial_overview": dict(report["actuarial_overview"]),
        "numeric_summary": records(report["numeric_stats"]),
        "categorical_summary": records(report["categorical_stats"]),
        "strongest_associations": records(top),
        "segment_examples": {
            name: records(table)
            for name, table in list(report["segments"].items())[:max_items]
        },
        "temporal": {
            name: records(table) for name, table in list(report["temporal"].items())[:max_items]
        },
        "warnings": list(report["warnings"])[:max_items],
        "methodological_note": (
            "Des associations descriptives ne démontrent pas de causalité. "
            "Ne jamais inventer de chiffres, de variables ou d'explications. "
            "Les taux actuariels exigent des unités et des périodes comparables."
        ),
    }
