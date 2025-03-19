writer_guidelines = """
 - Structure the section with clear headings for each sub-section, presenting the data and analysis in a concise and informative manner.
 - Prioritize clarity, factual accuracy, and a compelling presentation. Eliminate any uncertainty or ambiguity.
 - Maintain an objective, data-driven tone throughout the section.
 - Avoid technical jargon and use clear, concise language that is accessible to a wide range of readers.
 - Acknowledge both positive and negative financial indicators for a balanced assessment.
 - Base all content solely on the provided financial data and established financial principles. Do not introduce external information or speculate.
 - Explain complex financial concepts in plain language, ensuring accessibility for readers without a finance background.
 - Provide clear context for all metrics, explaining their significance (e.g., why a specific ratio is strong or concerning).
 - For longer sub-sections, use concise transition statements to link subtopics and maintain a strong narrative flow.
 - When presenting large datasets or multiple data points, prioritize clear and concise formatting.
 - Do not use bullet points for sub-section headers.
 - Use bullet points for lists of related items.
 - Emphasize important sub-section headers and key metrics by using bold text.
 - Format the output as markdown without code blocks.
 - Make sure that special characters such as '$', '%', and '&' are not escaped.
 - Do not include the date in the section heading or as a separate subheading.
"""

critic_guidelines = """
 - Identify any factual errors, inconsistencies, or logical gaps.
 - Assess the clarity and effectiveness of the writing. Are concepts explained clearly and concisely?
 - Evaluate whether all statements and claims are adequately supported by the provided data. Does the analysis offer meaningful, data-driven insights for investors?
 - Evaluate the coherence of the narrative. Does the section flow logically and tell a compelling story?
 - Check for completeness. Are all key aspects of the topic covered adequately?
 - Evaluate the effectiveness of data presentation, including the use of tables and bullet points. Are they used appropriately and effectively to enhance understanding?
 - If data is missing, it's usually better to suggest editing it out rather than drawing attention to its absence.
 - Provide specific, actionable feedback for improvement, referencing line numbers or specific phrases where possible.
 - Ensure the section maintains an objective, data-driven tone and avoids speculation.
 - Do not suggest adding charts, graphs or imagery.
 - You must output the full, verbatim draft section content that you have critiqued as well as your feedback (in the Pydantic model).
"""

editor_guidelines = """
 - Focus on enhancing the content in line with the feedback and avoid cutting content unless it's necessary.
 - Achieve exceptional clarity and accuracy while maintaining conciseness.
 - Use only the available information provided to you; do not introduce new data or suggest further research.
 - If data is missing, rewrite to maintain a strong narrative without it, rather than highlighting its absence.
 - Provide insightful context for all metrics, explaining their implications. Do not assume that the reader has the same knowledge as you.
 - Explain complex concepts in plain language, ensuring accessibility for readers without a finance background.
 - Acknowledge both positive and negative indicators for a balanced assessment.
 - When presenting large datasets or multiple data points, prioritize clear and concise formatting.
 - Express percentages with two decimal places for precision.
 - Use markdown formatting for all tables, ensuring concise and informative headings.
 - Make sure that special characters such as '$', '%', and '&' are not escaped.
 - Use bold text to highlight important sub-section headers and key metrics.
 - Do not use bullet points for sub-section headers.
 - Use bullet points for lists of related items.
 - Ensure consistent labeling of timeframes for all datasets (e.g., quarterly, yearly).
 - For longer sub-sections, use concise transition statements to link subtopics and maintain a strong narrative flow.
 - Avoid unnecessary disclaimers or caveats.
 - Do not explicitly mention the critic's feedback in the revised text.
 - Do not include the publication date in the section heading or as a separate subheading.
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