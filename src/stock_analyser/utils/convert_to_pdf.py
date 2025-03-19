from md2pdf.core import md2pdf
from pathlib import Path
from stock_analyser.utils.constants import OUTPUT_DIR, CSS_FILE

def convert_md_to_pdf(md_file_path: Path) -> Path:
    css_file = CSS_FILE
    plots_base_url = OUTPUT_DIR / 'plots'

    pdf_file_path = Path(md_file_path).with_suffix('.pdf')

    md2pdf(pdf_file_path,
           md_file_path=md_file_path,
           base_url=plots_base_url,
           css_file_path=css_file
    )

    return pdf_file_path


if __name__ == "__main__":
    md = '/home/j/ai/crewAI/finance/stock_analyser/final_reports/2025-03-18/NVDA_Stock_Analysis_Report_20250318_224327.md'
    pdf = convert_md_to_pdf(md)
    print(f"PDF file saved to {pdf}")