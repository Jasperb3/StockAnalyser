import textwrap
from typing import List
import matplotlib.pyplot as plt
from palettable.cartocolors.sequential import SunsetDark_7
from stock_analyser.utils.models import RiskSeverity


def plot_risk_severity(risk_severity: List[RiskSeverity], output_dir: str, timestamp: str):

    # Wrap long risk names, then replace spaces with newlines.
    wrapped_risks = ['\n'.join(textwrap.wrap(risk.risk, width=20, break_long_words=False)) for risk in risk_severity]
    severity = [risk.severity for risk in risk_severity]


    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust figure size as needed
    bars = ax.bar(wrapped_risks, severity, facecolor=SunsetDark_7.mpl_colors, edgecolor='black')

    # Dynamic font size (adjust the formula as needed)
    max_risk_length = max(len(risk) for risk in wrapped_risks)
    num_risks = len(risk_severity)
    fontsize = max(8, 16 - num_risks * 0.2 - max_risk_length * 0.1)
    
    ax.set_title("Risk Severity Assessment", fontsize=fontsize+4)
    ax.set_xlabel("Identified Risks", fontsize=fontsize+2, labelpad=10)
    ax.set_ylabel("Severity", fontsize=fontsize+2)

    # Rotate x-axis labels AND set the correct alignment
    plt.xticks(rotation=33.75, ha="right", fontsize=fontsize)

    plt.tight_layout()

    output_path = f"{output_dir}/risk_severity_{timestamp}.png"
    plt.savefig(output_path)

    return output_path


if __name__ == "__main__":
    # Create RiskSeverity objects for the example
    risk_severity = [
        RiskSeverity(risk="International Market Volatility Risk", severity=8),
        RiskSeverity(risk="National Regulatory Hurdles Risk", severity=7),
        RiskSeverity(risk="Company-specific Risk Possibilities", severity=6),
        RiskSeverity(risk="Non-Existant but Real Geopolitical Tensions Risk", severity=7),
        RiskSeverity(risk="National Market Volatility Risk", severity=3),
        RiskSeverity(risk="International Regulatory Hurdles Risk", severity=10),
        RiskSeverity(risk="Company-non-specific Risk Possibilities", severity=6),
        RiskSeverity(risk="Existant but Unreal Geopolitical Tensions Risk", severity=7)
    ]

    output_dir = "/home/j/ai/crewAI/finance/stock_analyser/"  # Use a valid path
    timestamp = "20210101_120000"

    plot_risk_severity(risk_severity, output_dir, timestamp)
