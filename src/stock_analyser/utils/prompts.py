data_research_guidelines = """
1. Monetary Values (Absolute):
    * Currency Symbol:
        * Use the appropriate currency symbol. Place it before the first digit.
        * Examples: $1,234.56; €1,234.56; £1,234.56; ¥1,234.56
    * Negative Sign:
        * Place the negative sign directly before the currency symbol.
        * Example: -$1,234.56
    * Thousand Separator:
        * Use a comma (,) as the thousand separator.
        * Example: $1,234,567.89 (US)
    * Decimal Separator:
        * Use a period (.) as the decimal separator.
        * Example: $1,234.56
    * Large Numbers:
        * For values ≥ 1,000,000, use a combination of numbers and words.
        * Use up to three decimal places for the numerical portion.
        * Examples:
            * $1,234,567: $1.235 million
            * $1,234,567,890: $1.235 billion
            * $1,234,567,890,123: $1.235 trillion
        * For values < 1,000,000:
            * Use thousand separators.
            * Example: $987,654.32
        * For values < 1,000:
            * Display the number as is (unless a specific level of precision is required).
            * Example: $789
    * Zero Values:
        * Display as "$0.00" (or the appropriate currency and decimal places) or simply "0" if the context is clear.
    * Parentheses for negative numbers:
        * Do not enclose negative numbers in parentheses. Instead use a negative sign, as outlined above.
2. Percentages:
    * Percentage Symbol:
        * Use the percentage symbol (%) after the number, with no space.
        * Example: 25.5%
    * Decimal Places:
        * Generally, use one or two decimal places. The appropriate level of precision depends on the context.
        * For very small percentages, or when showing changes in percentages, more decimal places may be necessary.
        * Examples: 12.3%; 12.34%; 0.05%
    * Negative Percentages:
        * Use a negative sign before the number.
        * Example: -5.2%
3. Shares and Units:
    * Whole Numbers:
        * Express the number of shares or units as a whole number, unless fractional shares or units are specifically relevant.
        * Example: 1,234,567 shares
    * Thousand Separator:
        * Use thousand separators for large numbers of shares.
        * Example: 1,234,567 shares
    * Labeling:
        * Clearly label what the number represents (e.g., "shares outstanding," "units sold").
4. Per-Share Data:
    * Currency and "Per Share" Label:
        * Include the currency symbol and the phrase "per share."
        * Example: $10.50 per share
    * Decimal Places:
        * Use a consistent number of decimal places, typically two or three.
        * Example: $10.50 per share; $2.345 per share
5. Ratios and Multiples:
    * Decimal Places:
        * Use a consistent number of decimal places (typically one or two) appropriate to the ratio.
        * Examples: 2.5; 1.25
    * Labeling:
        * Clearly label the ratio (e.g., "Debt-to-Equity Ratio," "Price-Earnings Ratio").
    * "Times" Notation:
        * For ratios expressed as a number of times, you use "x".
        * Examples: 2.5x; 1.25x
6. Dates:
    * Use ISO 8601 date format throughout the report.
    * Example: 2024-03-31
7. Missing Data:
    * When data is not available, use "N/A"
    * Example: N/A 
"""


writer_guidelines = """
 - Organize the section clearly using nested markdown headings (H3, H4, H5).
 - Use concise transitions to link subtopics and ensure a cohesive narrative flow.
 - Present data and analysis clearly, concisely, and informatively.
 - Adopt an informative, objective, data-driven, and sophisticated tone suited to a well-educated and curious audience.
 - Prioritize clarity, factual accuracy, and compelling content presentation.
 - Project expertise, authority, and confidence; eliminate uncertainty or ambiguity.
 - Avoid technical jargon; use clear language accessible to diverse readers.
 - Provide a balanced assessment by acknowledging both positive and negative financial indicators.
 - Rely solely on provided financial data and established financial principles; avoid external information and speculation.
 - Clearly explain complex financial concepts in plain language, ensuring accessibility to non-finance readers.
 - Offer context for all metrics, clearly outlining their significance and implications.
 - For longer subsections, combine paragraphs, lists, and tables to enhance readability.
 - Use bullet points for concise presentation of large datasets, related item lists, or multiple data points.
 - Highlight important subsection headers and key metrics using bold text.
 - Format content using markdown; avoid code blocks.
 - Ensure special characters like '$', '%', and '&' are not escaped.
 - For amounts equal to or exceeding one million, express figures with up to three decimal places, followed by the appropriate word ('million', 'billion', 'trillion', etc.), e.g. $1.234 billion.
 - Do not include the current date in the section heading or as a separate subheading.
"""

critic_guidelines = """
 - Identify factual errors, inconsistencies, or logical gaps.
 - Evaluate clarity and effectiveness: Are concepts explained clearly and concisely?
 - Confirm statements and claims are supported by provided data; ensure analysis provides meaningful, data-driven insights for investors.
 - Assess narrative coherence: Does the content flow logically and tell a compelling story?
 - Check for completeness: Are all key aspects adequately covered?
 - Evaluate the effectiveness of data presentation, including subsection headers and bullet points. Are these elements used effectively to enhance understanding?
 - Suggest removing incomplete or missing data rather than highlighting its absence.
 - Provide specific, actionable feedback, referencing particular content or phrases where possible.
 - Confirm the section maintains an objective, data-driven tone without speculation.
 - Do not suggest the inclusion of charts, graphs, or imagery.
 - Include the full, verbatim draft section along with your feedback (in the Pydantic model).
"""

editor_guidelines = """
 - Enhance content clarity and accuracy based strictly on provided feedback; minimize content cuts unless necessary.
 - Ensure exceptional clarity, accuracy, and conciseness.
 - Ensure the section is well-organized using nested markdown headings (H3, H4, H5).
 - Utilize only the available information; avoid introducing new data or recommending additional research.
 - Rewrite sections to maintain narrative strength when data is missing rather than highlighting its absence.
 - Provide insightful context for all metrics, clearly explaining their implications without assuming prior reader knowledge.
 - Clearly explain complex financial concepts in plain language, ensuring accessibility to non-finance readers.
 - Present a balanced assessment by acknowledging both positive and negative indicators.
 - Clearly and concisely format large datasets or multiple data points.
 - Express percentages with two decimal places for precision.
 - Format all tables in markdown with concise, informative headings.
 - Ensure special characters like '$', '%', and '&' are not escaped.
 - Use bold text to highlight important subsection headers and key metrics.
 - Use bullet points for concise presentation of large datasets, related item lists, or multiple data points.
 - For amounts equal to or exceeding one million, express figures with up to three decimal places, followed by the appropriate word ('million', 'billion', 'trillion', etc.), e.g. $1.234 billion.
 - Ensure consistent labeling of dataset timeframes (e.g., quarterly, yearly).
 - Use concise transitions in longer subsections to maintain a cohesive narrative flow.
 - Avoid unnecessary disclaimers or caveats.
 - Do not explicitly mention the critic's feedback in revised text.
 - Do not include the current date in the section heading or as a separate subheading.
"""


expert_analyst_writer_prompt = """
 - The section must be formatted in Markdown without code blocks.
 - Use quotation marks and italics for direct quotes.
 - Base the investment signal entirely on the the analysis provided to you.
 - Ensure clarity and precision. Avoid vague language or ambiguity.
 - Present information factually and in an easily digestible manner.
 - Do not include any disclaimers or calls for additional research.
 - Do not include a publication date as a separate subheading.
"""