import os
import json

from dotenv import load_dotenv
from openai import OpenAI

from tools import (
    load_and_validate_data,
    portfolio_summary,
    claim_frequency,
    frequency_by_segment,
    risk_segments,
    detect_anomalies,
    filtered_frequency
)

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# Chargement du portefeuille
df = load_and_validate_data(
    "data/sample_portfolio.csv"
)


# Description des outils accessibles au LLM
TOOLS = [
    {
        "type": "function",
        "name": "portfolio_summary",
        "description": (
            "Retourne un résumé global du portefeuille : "
            "nombre de contrats, exposition totale, nombre de sinistres "
            "et fréquence globale."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False
        },
        "strict": True
    },

    {
        "type": "function",
        "name": "claim_frequency",
        "description": (
            "Calcule la fréquence globale de sinistres du portefeuille."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False
        },
        "strict": True
    },

    {
        "type": "function",
        "name": "frequency_by_segment",
        "description": (
            "Calcule la fréquence de sinistres par segment "
            "pour une variable du portefeuille."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "variable": {
                    "type": "string",
                    "description": (
                        "Variable utilisée pour segmenter le portefeuille, "
                        "par exemple region, vehicle_brand, vehicle_fuel ou area."
                    )
                },
                "min_exposure": {
                    "type": "number",
                    "description": "Exposition minimale requise pour conserver un segment."
                }
            },
            "required": ["variable", "min_exposure"],
            "additionalProperties": False
        },
        "strict": True
    },

    {
        "type": "function",
        "name": "risk_segments",
        "description": (
            "Identifie les segments ayant les fréquences de sinistres "
            "observées les plus élevées."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "variable": {
                    "type": "string",
                    "description": (
                        "Variable de segmentation, par exemple region, "
                        "vehicle_brand, vehicle_fuel ou area."
                    )
                },
                "top_n": {
                    "type": "integer",
                    "description": "Nombre de segments à retourner."
                },
                "min_exposure": {
                    "type": "number",
                    "description": "Exposition minimale requise pour conserver un segment."
                }
            },
            "required": [
                "variable",
                "top_n",
                "min_exposure"
            ],
            "additionalProperties": False
        },
        "strict": True
    },

    {
    "type": "function",
    "name": "detect_anomalies",
    "description": (
        "Analyse la qualité des données du portefeuille et détecte "
        "les principales anomalies : valeurs manquantes, expositions "
        "invalides, nombres de sinistres négatifs et identifiants "
        "de contrats dupliqués."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "additionalProperties": False
    },
    "strict": True
    },

    {
    "type": "function",
    "name": "filtered_frequency",
    "description": (
        "Calcule la fréquence de sinistres pour une sous-population "
        "du portefeuille définie par une condition. "
        "Utilise cet outil pour des questions comme la fréquence des "
        "conducteurs de moins de 30 ans ou des véhicules diesel."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "variable": {
                "type": "string",
                "description": (
                    "Variable sur laquelle appliquer le filtre, par exemple "
                    "driver_age, vehicle_age, vehicle_fuel ou region."
                )
            },
            "operator": {
                "type": "string",
                "enum": ["==", "!=", "<", "<=", ">", ">="],
                "description": "Opérateur utilisé pour le filtre."
            },
            "value": {
                "description": "Valeur utilisée pour le filtre.",
                "anyOf": [
                    {"type": "number"},
                    {"type": "string"}
                ]
            }
        },
        "required": [
            "variable",
            "operator",
            "value"
        ],
        "additionalProperties": False
    },
    "strict": True
}
]

def execute_tool(tool_name, arguments):
    """
    Exécute l'outil Python demandé par le LLM
    et retourne son résultat.
    """

    if tool_name == "portfolio_summary":
        return portfolio_summary(df)

    elif tool_name == "claim_frequency":
        return claim_frequency(df)

    elif tool_name == "frequency_by_segment":
        return frequency_by_segment(
            df,
            variable=arguments["variable"],
            min_exposure=arguments["min_exposure"]
        ).to_dict(orient="records")

    elif tool_name == "risk_segments":
        return risk_segments(
            df,
            variable=arguments["variable"],
            top_n=arguments["top_n"],
            min_exposure=arguments["min_exposure"]
        ).to_dict(orient="records")

    elif tool_name == "detect_anomalies":
        return detect_anomalies(df)

    elif tool_name == "filtered_frequency":
        return filtered_frequency(
            df,
            variable=arguments["variable"],
            operator=arguments["operator"],
            value=arguments["value"]
        )

    else:
        raise ValueError(
            f"Outil inconnu : {tool_name}"
        )
    
def ask_agent(question):
    """
    Envoie une question à l'agent actuariel.
    Le LLM choisit l'outil actuariel approprié si nécessaire.
    """

    # Premier appel : le LLM analyse la question
    # et décide éventuellement d'utiliser un outil.
    response = client.responses.create(
        model="gpt-5.6-luna",
        instructions=(
            "Tu es un assistant spécialisé en actuariat IARD. "
            "Pour toute question quantitative concernant le portefeuille, "
            "tu dois utiliser les outils disponibles. "
            "N'invente jamais de résultats numériques."
        ),
        input=question,
        tools=TOOLS
    )

    # Recherche d'un appel d'outil demandé par le LLM
    for item in response.output:

        if item.type == "function_call":

            # Arguments choisis par le LLM
            arguments = json.loads(item.arguments)

            print(
                f"[DEBUG] Outil choisi : {item.name} | "
                f"Arguments : {arguments}"
            )

            # Exécution de l'outil Python
            result = execute_tool(
                item.name,
                arguments
            )

            # Conversion du résultat en JSON
            tool_output = json.dumps(
                result,
                default=lambda x: (
                    x.item()
                    if hasattr(x, "item")
                    else str(x)
                )
            )

            # Deuxième appel :
            # le résultat Python est renvoyé au LLM
            # afin qu'il l'interprète.
            final_response = client.responses.create(
                model="gpt-5.6-luna",
                instructions=(
                    "Tu es un assistant spécialisé en actuariat IARD. "
                    "Interprète les résultats des outils de manière concise "
                    "et rigoureuse. "
                    "Ne présente jamais une fréquence élevée comme une preuve "
                    "de risque intrinsèque ou causal."
                ),
                previous_response_id=response.id,
                input=[
                    {
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": tool_output
                    }
                ],
                tools=TOOLS
            )

            return final_response.output_text

    # Si aucun outil n'est nécessaire,
    # on retourne directement la réponse du LLM.
    return response.output_text

if __name__ == "__main__":

    answer = ask_agent(
        "Quelle est la fréquence de sinistres "
        "pour les conducteurs de moins de 30 ans ?"
    )

    print(answer)
