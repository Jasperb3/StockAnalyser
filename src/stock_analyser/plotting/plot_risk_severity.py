import textwrap
from typing import List
import matplotlib.pyplot as plt
from palettable.cartocolors.sequential import SunsetDark_7
from stock_analyser.utils.models import RiskSeverity
from stock_analyser.utils.constants import FONT_FAMILY

def plot_risk_severity(risk_severity: List[RiskSeverity], output_dir: str, timestamp: str):

    # Wrap long risk names, then replace spaces with newlines.
    wrapped_risks = ['\n'.join(textwrap.wrap(risk.risk, width=20, break_long_words=False)) for risk in risk_severity]
    severity = [risk.severity for risk in risk_severity]

    # Define the color palette
    colors = ["#fcde9c", "#faa476", "#f0746e", "#e34f6f", "#dc3977", "#b9257a", "#7c1d6f"]
    num_colors = len(colors)

    # Map severity to color index
    def get_color(sev):
        # Severity scale: 1-10
        # Grouping: 0, 1, 2, 3 (lightest) - 9, 10 (darkest)
        if sev <= 3:
            color_index = 0
        elif sev >= 9:
            color_index = num_colors - 1
        else:
            # Map 4-8 to the remaining colors (index 1 to 5)
            # Normalize to 0-1 range within 4-8
            normalized_severity = (sev - 4) / (8 - 4 + 1e-9)  # Add small value
            color_index = int(normalized_severity * (num_colors - 2)) + 1

        return colors[color_index]

    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust figure size as needed

    # Apply color mapping to each bar
    bar_colors = [get_color(s) for s in severity]
    bars = ax.bar(wrapped_risks, severity, facecolor=bar_colors, edgecolor='black')

    # Dynamic font size (adjust the formula as needed)
    max_risk_length = max(len(risk) for risk in wrapped_risks)
    num_risks = len(risk_severity)
    fontsize = max(8, 16 - num_risks * 0.2 - max_risk_length * 0.1)
    
    ax.set_title("Risk Severity Assessment", fontsize=fontsize+4, fontfamily=FONT_FAMILY)
    ax.set_xlabel("Identified Risks", fontsize=fontsize+2, labelpad=10, fontfamily=FONT_FAMILY)
    ax.set_ylabel("Severity", fontsize=fontsize+2, fontfamily=FONT_FAMILY)

    # Rotate x-axis labels AND set the correct alignment
    plt.xticks(rotation=33.75, ha="right", fontsize=fontsize)

    plt.tight_layout()

    output_path = f"{output_dir}/risk_severity_{timestamp}.png"
    plt.savefig(output_path, dpi=300)

    return output_path


if __name__ == "__main__":
    # Create RiskSeverity objects for the example
    risk_severity = [
        RiskSeverity(risk="International Market Volatility Risk", severity=8),
        RiskSeverity(risk="National Regulatory Hurdles Risk", severity=9),
        RiskSeverity(risk="Company-specific Risk Possibilities", severity=6),
        RiskSeverity(risk="Non-Existant but Real Geopolitical Tensions Risk", severity=7),
        RiskSeverity(risk="National Market Volatility Risk", severity=3),
        RiskSeverity(risk="International Regulatory Hurdles Risk", severity=10),
        RiskSeverity(risk="Company-non-specific Risk Possibilities", severity=2),
        RiskSeverity(risk="Existant but Unreal Geopolitical Tensions Risk", severity=4)
    ]

    output_dir = "/home/j/ai/crewAI/finance/stock_analyser/"  # Use a valid path
    timestamp = "20210101_120000"

    plot_risk_severity(risk_severity, output_dir, timestamp)
    print(f"Plot saved to: {output_dir}/risk_severity_{timestamp}.png")
