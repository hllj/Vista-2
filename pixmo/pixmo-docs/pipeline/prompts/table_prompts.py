NUM_TOPICS = 5




GENERATE_TABLE_TOPICS_PROMPT = """You are an expert in data analysis and have a broad knowledge of different topics.
My persona is: "{persona}"
I want you to generate {num_topics} topics for {figure_type} that I will be interested in or I may see during my daily life given my persona.

Here are the requirements:
1. Each topic is a high-level summary of tabular data, e.g., "Sales data of a grocery store in 2021 Q1".
2. The topics should be diverse to help me generate varied tables. Each topic should be unique and not overlap with others.
3. The topics are conditioned on the table type. Please ensure the topics you provided can be best visualized in "{figure_type}".
4. All topics must be in English, even if the persona is non-English.
5. List {num_topics} topics for "{persona}" and separate them with a | character, e.g., topic1 | topic2 | ...... | topic{num_topics}.
Do not include any additional text at the beginning or end of your response."""




GENERATE_TABLE_DATA_PROMPT = """You are an expert in data analysis and have broad knowledge about various topics.
My persona is: "{persona}"
I need some data about "{topic}", which can be used to generate a {figure_type}. 
Here are the requirements:
1. The data structure must be suitable for the {figure_type}.
2. The contents are related to the topic and customized according to my persona.
3. The data should be realistic, and the contents should be named using real-world entities. Do not use placeholder names like xxA, xxB, etc.
4. The data should be diverse and contain multiple data points to ensure the table is informative.
5. Do not provide too much data. Just necessary data points to satisfy the topic and figure type.
6. All data must be in English, even if the persona is non-English.
Please provide the data in CSV format without additional text at the beginning or end."""




GENERATE_TABLE_DATA_JSON_PROMPT = """You are an expert in data analysis and have broad knowledge about various topics.
My persona is: "{persona}"
I need some data about "{topic}", which can be used to generate a {figure_type}. 
Here are the requirements:
1. The data structure must be suitable for the {figure_type}.
2. The contents are related to the topic and customized according to my persona.
3. The data should be realistic, and the contents should be named using real-world entities. Do not use placeholder names like xxA, xxB, etc.
4. The data should be diverse and contain multiple data points to ensure the table is informative.
5. Do not provide too much data. Just necessary data points to satisfy the topic and figure type.
6. All data must be in English, even if the persona is non-English.
Please provide the data in JSON format without additional text at the beginning or end."""




GENERATE_TABLE_QA_PROMPT = """You are an expert in data analysis and good at asking questions about tables.
My persona is: "{persona}"
I want you to generate some question-answer pairs of a table about {topic}, which I would ask.
Instead of showing the visual table, I provide the data and the code that generates the table.

Here is the data:
{data}

Here is the code that generates the table:
```
{code}
```

Please come up with a list of *reasonable questions* that people will ask when they see the rendered table. Here are the requirements:
1. **Question Style**: The questions must be natural and related to the table, which can help interpret the data and understand the insights.
    (1) The questions vary in complexity. Some are easy to answer by just referring to the table, and some are challenging and require multiple-step reasoning.
    (2) The questions should be answerable based on the *visual information* in the table. Don't include any coding details in the questions since this type of information is not visible in the rendered table.

2. **Question Types**: Most questions are short-answer, but some can be multiple-choice, yes/no, or summary questions. You can use the following types:
    (1) Short-answer: At least 5 short-answer questions.
    (2) Multiple-choice: There should be at least two multiple-choice questions. The number of options can be 3, 4, 5, or more. The option labels can be different types: alphabet, Arabic numerals, or Roman numerals. The correct option should be different in each question.
    (3) Yes/No (True/False): At least 1 binary question.
    (4) Summary: At least 1 summary question that asks for describing the *entire table* or writing a caption for the table.
    (5) Unanswerable: At least 1 question cannot be answered based on the visual information in the table. The answer to this question can be "Cannot be determined", "Not enough information", "I don't know", etc.

3. **Provide Explanations**: In addition to a *concise answer* for each question, provide an explanation that details the reasoning steps to reach the answer. For the summary question, the explanation is a more detailed description of the table.

4. **Response Format**: separate the question, answer, and explanation with a | character: question | answer | explanation. The question-answer pairs should be separated by double newlines (\n\n).
For example:
what's the difference between the max value of column A and column B? | 15 | Max of A is 10 and that of B is 25, so the difference is 25-10=15

Which group has the highest value? A. Group-1 B. Group-2 C. Group-3 | B | Group-1 is 20, Group-2 is 25, and Group-3 is 15, so Group-2 has the highest value.

... ...

Do not include any additional text at the beginning or end of your response."""




GENERATE_TABLE_QA_SHORT_ANSWER_PROMPT = """You are an expert in data analysis and good at asking questions about tables.
My persona is: "{persona}"
I want you to generate some question-answer pairs of a "{figure_type}" about "{topic}", which I would ask.
Instead of showing the table, I provide the data and the code that generates the table.

<data>
{data}
</data>

<code>
{code}
</code>

Please come up with a list of *reasonable questions* that people will ask when they see the rendered table. Here are the requirements:
1. **Question Style**: The questions must be natural and related to the table, which can help interpret the data and understand the insights.
    (1) The questions vary in complexity. Some are easy to answer by just referring to the table, and some are challenging and require multiple-step reasoning.
    (2) The questions should be answerable based on the *visual information* in the table. Don't include any coding details in the questions since this type of information is not visible in the figure.

2. **Question Types**: All questions should be short-answer questions that are answerable based on the visual information in the table.
    (1) **Retrieval questions** ask for specific values or facts in the table (keep the exact same format in the table). Some easy questions can be answered directly from the table. Some more challenging questions may require combining and filtering information from multiple rows or columns.
    (2) **Compositional questions** contain multiple mathematical/logical operations like sum, difference, and average, median, etc.
    (3) **Fact-based questions** ask for whether a specific fact is yes or no (true of false) based on the data. Make sure to have balanced numbers of yes/no questions.
    (4) **Complex reasoning questions** require multiple steps reasoning over multiple data points in the table. These questions should be more challenging and require a deeper understanding of the data.

3. **Provide Explanations**: 
    (1) In addition to a *concise answer* (as short as possible) for each question, provide a *step-by-step explanation* that details the reasoning steps to reach the answer.
    (2) For complex reasoning questions, the explanation should be more detailed and include all the necessary steps.

4. **Response Format**: Use | to separate the question, explanation, and concise answer for each example. 
    (1) Follow this format: question | explanation | concise answer, e.g., what's the average value of group A? | There are 5 data points in group A, sum them up is (34 + 45 + 23 + 56 + 12) = 170, so the average is 170/5 = 34 | 34
    (2) Separate the question-answer pairs by double newlines (\n\n). question1 | explanation1 | answer1\n\nquestion2 | explanation2 | answer2\n\n...
    (3) Do not provide too many questions, 5-10 questions are enough. Focus on the diversity and quality of the questions with a very detailed explanation for challenging questions.
    (4) The concise answer should be as short as possible and directly answer the question. All words in the response should be processed in natural language, no coding terms and remove any unnecessary characters.

Please follow the format strictly and do not include any additional text at the beginning or end of your response."""




GENERATE_TABLE_CODE_LATEX_PROMPT = """You are an expert in data analysis and good at writing LaTeX code to generate tables.
My persona is: "{persona}"
I have some data about {topic} which can be used to generate a {figure_type}.

Here is the data (JSON format):
<data>
{data}
</data>

Please write a LaTeX script to generate a {figure_type} using the data provided. Here are the requirements:
1. **Style Requirements**:
    (1) Try to be creative and change the default arguments (e.g., font, color, border, shade, etc) to make the table style unique.
    (2) Select appropriate design, layout and margin, ensuring the table is saved with all the elements visible, clear and easy to understand, with no text overlapping, etc.
    (3) Do not use white text color or other light colors on a white background, e.g., when row color is white, use a dark text color.

2. **Code Requirements**:
    (1) You need to hardcode the provided data into the LaTeX script to generate the table. Be careful with the syntax and formatting of the LaTeX script.
    (2) Use `standalone` LaTex document class to generate the table and add some border margin (`[border=xxpt]`). **Do not add the page number.**

3. **Output Requirements**:
    Put ```latex at the beginning and ``` at the end of the script to separate the code from the text. This will help me easily extract the code.

Please don't answer with any additional text in the script. Your whole response should be the LaTex code, which can be directly executed."""




GENERATE_TABLE_CODE_MATPLOTLIB_PROMPT = """You are an expert in data analysis and good at writing code (Python `pandas`) to generate tables.
My persona is: "{persona}"
I have some data about {topic} which can be used to generate a {figure_type}.

Here is the data (CSV format, already loaded as a pd.DataFrame):
<data>
{data}
</data>

Please define a Python function (using `pandas`) called `generate_table` that generates a {figure_type} using the data provided. Here are the requirements:
1. **Style Requirements**:
    (1) Try to be creative and change the default arguments (e.g., font, color, border, shade, etc) to make the table style unique. Set the style by using `plt.style.use('{style_name}')`.
    (2) Select the appropriate design scale (e.g., column width) to ensure the information in each cell is clear and easy to understand, with no text overlapping, etc.

2. **Code Requirements**: create a function called `generate_table` that generates the table using `pandas`.
    (1) The data, which is loaded as a pd.DataFrame is provided as the first argument for the function. The function has no other arguments. You may need to adjust the data format based on the requirements and your design.
    (2) Remember to import necessary libraries (e.g., `import numpy as np`, `import pandas as pd`, `import matplotlib.pyplot as plt`, etc) at the beginning of the script.
    (3) The `generate_table` function should save the table to a BytesIO and then return the table as a PIL Image object. **Do not close the BytesIO object.**
    (4) Select appropriate margins, resolution, and tight layout, ensuring the table is saved with all the elements visible.
    (5) Only define the function and do not call it. Do not show the table. No need to show example usage.

3. **Output Requirements**:
    Put ```python at the beginning and ``` at the end of the script to separate the code from the text. This will help me easily extract the code.

Please don't answer with any additional text in the script. Your whole response should be the Python code, which can be directly executed."""




GENERATE_TABLE_CODE_PLOTLY_PROMPT = """You are an expert in data analysis and good at writing code (Python `plotly`) to generate tables.
My persona is: "{persona}"
I have some data about {topic} which can be used to generate a {figure_type}.

Here is the data (CSV format, already loaded as a pd.DataFrame):
<data>
{data}
</data>

Please define a Python function (using `plotly`) called `generate_table` that generates a {figure_type} using the data provided. Here are the requirements:
1. **Style Requirements**:
    (1) Try to be creative and change the default arguments (e.g., font, color, border, shade, etc) to make the table style unique.
    (2) Select the appropriate design scale (e.g., column width) to ensure the information in each cell is clear and easy to understand, with no text overlapping, etc.

2. **Code Requirements**: create a function called `generate_table` that generates the table using `plotly`. Do not use matplotlib or other libraries.
    (1) The data, which is loaded as a pd.DataFrame is provided as the first argument for the function. The function has no other arguments. You may need to adjust the data format to fit the `plotly` specification.
    (2) Remember to import necessary libraries (e.g., `import numpy as np`, `import plotly.express as px`) at the beginning of the script.
    (3) The `generate_table` function should save the table to a BytesIO and then return the table as a PIL Image object. **Do not close the BytesIO object.**
    (4) Select appropriate margins and tight layout, ensuring the table is saved with all the elements visible.
    (5) Only define the function and do not call it. Do not show the table. Save the table with enough resolution to be clearly visible. No need to show example usage.

3. **Output Requirements**:
    Put ```python at the beginning and ``` at the end of the script to separate the code from the text. This will help me easily extract the code.

Please don't answer with any additional text in the script. Your whole response should be the Python code, which can be directly executed."""




GENERATE_TABLE_CODE_HTML_PROMPT = """You are an expert web designer and are good at writing HTML to create tables.
My persona is: "{persona}"
I have some data about {topic} which can be used to generate a {figure_type}.

Here are the materials (JSON format):
<data>
{data}
</data>

Please use HTML and CSS to generate a {figure_type} using the data provided. Here are the requirements:
1. **Style Requirements**: Feel free to use any CSS framework, libraries, JavaScript plugins, or other tools to create the table.
    (1) Try to be creative and make the web page style, fonts, colors, borders and visual layout unique with CSS. Taking persona, topic, and table type into consideration when designing the table.
    (2) Select the appropriate design scale (e.g., collumn width, cell size, margins, etc) to ensure the information in the table is clear and easy to understand, with no text overlapping etc.

2. **Code Requirements**: 
    (1) You need to hardcode the provided data into the HTML script to generate the document. Be careful with the syntax and formatting of the HTML script.
    (2) Put everything in one HTML file. Do not use external CSS or JavaScript files.

3. **Output Requirements**:
    Put ```html at the beginning and ``` at the end of the script to separate the code from the text.

Please don't answer with any additional text in the script, your whole response should be the HTML code which can be directly executed."""



TABLE_QUESTION_TYPES = [
    "Basic: Ask for basic visual elements in the table, such as the title, number of rows, columns, or cells.",
    "Basic: Ask for a specific data point or value in the table.",
    "Compositional: contains two mathematical/logical operations like sum, diference, average, median, etc.",
    "Compositional: contains three mathematical/logical operations like sum, diference, average, median, etc.",
    "Compositional: contains four mathematical/logical operations like sum, diference, average, median, etc.",
    "Compositional: contains five or more mathematical/logical operations like sum, diference, average, median, etc.",
    "Comparison: Ask about the relationship between two or more data points.",
    "Comparison: Ask about the relationship between three or more data points..",
    "Comparison: Ask about the relationship between four or more data points..",
    "Comparison: Ask about the relationship among all rows or columns in the table.",
    "Comparison: compare the data points under specific conditions.",
    "Data Retrieval: Ask for specific data points or values in the table.",
    "Data Retrieval: Ask for some extreme (highest/lowest) data points or values in the table.",
    "Data Retrieval: Ask for some non-extreme (somewhere in the middle) data points or values in the table.",
    "Data Retrieval: Ask for the data points or values that meet specific conditions.",
    "Data Retrieval: require combining and filtering information from multiple rows or columns.",
    "Data Retrieval: Ask for the critical points (e.g., turning points, intersection, changing trend, etc) in the table.",
    "Reasoning: Ask for a question requiring multi steps of reasoning to answer.",
    "Reasoning: Ask for a question requiring two steps of reasoning to answer.",
    "Reasoning: Ask for a question requiring three steps of reasoning to answer.",
    "Reasoning: Ask for a question requiring four steps of reasoning to answer.",
    "Reasoning: Ask for a question requiring five steps of reasoning to answer.",
    "Filtering: Ask for data points that meet specific conditions (like a SQL query).",
    "Filtering: query the data points that meet multiple conditions.",
    "Filtering: query the data points that meet two conditions.",
    "Filtering: query the data points that meet three conditions.",
    "Filtering: query the data points that meet four or more conditions.",
    "Structure: Ask about the structure of the data or the table.",
    "Counting: count the number of data points or groups or elements in the table.",
    "Counting: count the number of data points under specific conditions.",
    "Counting: count the number of data points under multiple conditions.",
    "Trend: Ask about the trend or pattern in the table.",
    "Fact Checking: given a statement, ask if it is true or false based on the table.",
    "Fact Checking: given a state with multiple conditions, answer with yes or no based on the table.",
    "Fact Checking: given a state with two conditions, answer with yes or no based on the table.",
    "Fact Checking: given a state with three conditions, answer with yes or no based on the table.",
    "Fact-based: Ask about the fact in the table, answer with yes or no.",
    "Yes/No: Ask a binary question that can be answered with yes or no.",
    "Yes/No: Ask a binary question that can be answered with yes or no with multi-step reasoning.",
    "Multiple Choice: Ask a multiple-choice question with hard negative options.",
    "Multiple Choice: Ask a multiple-choice question with 3-8 options.",
    "Multiple Choice: Ask a multiple-choice question with 4 options.",
    "Multiple Choice: Ask a multiple-choice question with more than 4 options.",
    "Long question: the question is long with detailed context and requires reasoning to answer.",
    "Short question: the question is short and can be answered directly by taking a glance at the table.",
    "Annotations: Ask about annotations in the table, such as text notes, highlights, or arrows pointing to specific parts.",
    "Table-Type Specific: Ask questions that are unique to this table type.",
    "Topic-Specific: Ask questions that are specific to the topic of the table.",
    ]




GENERATE_TABLE_QA_DIVERSE_ANSWER_PROMPT = """You are an expert in data analysis and good at asking questions about tables.
My persona is: "{persona}"
I want you to generate some question-answer pairs of a "{figure_type}" about "{topic}", which I would ask.
Instead of showing the table, I provide the data and the code that generates the table.

<data>
{data}
</data>

<code>
{code}
</code>

Please come up with a list of *reasonable questions* that people will ask when they see the rendered table. Here are the requirements:
1. **Question Style**: The questions must be natural and related to the table, which can help interpret the data and understand the insights.
    (1) The questions vary in complexity. Some are easy to answer by just referring to the table, and some are challenging and require multiple-step reasoning.
    (2) The questions should be answerable based on the *visual information* in the table. Don't include any coding details in the questions since this type of information is not visible in the figure.

2. **Question Types**: All questions should be short-answer questions that are answerable based on the visual information in the table.
    (1) **Retrieval questions** ask for specific values or facts in the table (keep the exact same format in the table). Some easy questions can be answered directly from the table. Some more challenging questions may require combining and filtering information from multiple rows or columns.
    (2) **Compositional questions** contain multiple mathematical/logical operations like sum, difference, and average, median, etc.
    (3) **Fact-based questions** ask for whether a specific fact is yes or no (true of false) based on the data. Make sure to have balanced numbers of yes/no questions.
    (4) **Complex reasoning questions** require multiple steps reasoning over multiple data points in the table. These questions should be more challenging and require a deeper understanding of the data.

3. **Other Question Types**:
    {selected_question_types}

4. **Provide Explanations**: 
    (1) In addition to a *concise answer* (as short as possible) for each question, provide a *step-by-step explanation* that details the reasoning steps to reach the answer.
    (2) For complex reasoning questions, the explanation should be more detailed and include all the necessary steps.

5. **Response Format**: Use | to separate the question, explanation, and concise answer for each example. 
    (1) Follow this format: question | explanation | concise answer, e.g., what's the average value of group A? | There are 5 data points in group A, sum them up is (34 + 45 + 23 + 56 + 12) = 170, so the average is 170/5 = 34 | 34
    (2) Separate the question-answer pairs by double newlines (\n\n). question1 | explanation1 | answer1\n\nquestion2 | explanation2 | answer2\n\n...
    (3) Do not provide too many questions, one for each question type is enough. Focus on the diversity and quality of the questions with a **very detailed explanation** for challenging questions.
    (4) The concise answer should be as short as possible and directly answer the question. All words in the response should be processed in natural language, no coding terms and remove any unnecessary characters.

Please follow the format strictly and do not include any additional text at the beginning or end of your response."""