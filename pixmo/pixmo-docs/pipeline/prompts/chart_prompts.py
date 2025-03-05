NUM_TOPICS = 5




GENERATE_CHART_TOPICS_PROMPT = """You are an expert in data analysis and have a broad knowledge of different topics.
My persona is: "{persona}"
I want you to generate {num_topics} topics for {figure_type} that I will be interested in or I may see during my daily life given my persona.

Here are the requirements:
1. Each topic is a high-level summary of statistical distribution with some details, e.g., "population growth in the US with a breakdown of the age groups."
2. The topics should be diverse to help me generate varied figures. Each topic should be unique and not overlap with others.
3. The topics are conditioned on the figure type. Please ensure the topics you provided can be best visualized in "{figure_type}".
4. All topics must be in English, even if the persona is non-English.
5. List {num_topics} topics for "{persona}" and separate them with a | character, e.g., topic1 | topic2 | ...... | topic{num_topics}.
Do not include any additional text at the beginning or end of your response."""




GENERATE_CHART_DATA_PROMPT = """You are an expert in data analysis and have broad knowledge about various topics.
My persona is: "{persona}"
I need some data about "{topic}", which can be used to generate a {figure_type}. 
Here are the requirements:
1. The data structure must be suitable for the {figure_type}.
2. The contents are related to the topic and customized according to my persona.
3. The data should be realistic, and the contents should be named using real-world entities. Do not use placeholder names like xxA, xxB, etc.
4. The data should be diverse and contain multiple data points to ensure the chart is informative.
5. Do not provide too much data. Just necessary data points to satisfy the topic and figure type.
6. All data must be in English, even if the persona is non-English.
Please provide the data in CSV format without additional text at the beginning or end."""




GENERATE_CHART_DATA_JSON_PROMPT = """You are an expert in data analysis and have broad knowledge about various topics.
My persona is: "{persona}"
I need some data about "{topic}", which can be used to generate a {figure_type}. 
Here are the requirements:
1. The data structure must be suitable for the {figure_type}.
2. The contents are related to the topic and customized according to my persona.
3. The data should be realistic, and the contents should be named using real-world entities. Do not use placeholder names like xxA, xxB, etc.
4. The data should be diverse and contain multiple data points to ensure the chart is informative.
5. Do not provide too much data. Just necessary data points to satisfy the topic and figure type.
6. All data must be in English, even if the persona is non-English.
Please provide the data in JSON format without additional text at the beginning or end."""




GENERATE_CHART_QA_PROMPT = """You are an expert in data analysis and good at asking questions about plots.
My persona is: "{persona}"
I want you to generate some question-answer pairs of a {figure_type} about {topic}, which I would ask.
Instead of showing the figure, I provide the data and the code that generates the figure.

Here is the data:
<data>
{data}
</data>

Here is the code that generates the figure:
<code>
{code}
</code>

Please come up with a list of *reasonable questions* that people will ask when they see the rendered figure. Here are the requirements:
1. **Question Style**: The questions must be natural and related to the figure, which can help interpret the data and understand the insights.
    (1) The questions vary in complexity. Some are easy to answer by just referring to the figure, and some are challenging and require multiple-step reasoning.
    (2) The questions should be answerable based on the *visual information* in the figure. Don't include any coding details in the questions since this type of information is not visible in the figure.

2. **Question Types**: Most questions are short-answer, but some can be multiple-choice, yes/no, or summary questions. You can use the following types:
    (1) Short-answer: At least 5 short-answer questions.
    (2) Multiple-choice: There should be at least two multiple-choice questions. The number of options can be 3, 4, 5, or more. The option labels can be different types: alphabet, Arabic numerals, or Roman numerals. The correct option should be different in each question.
    (3) Yes/No (True/False): At least 1 binary question.
    (4) Summary: At least 1 summary question that asks for describing the *entire figure* or writing a caption for the figure.
    (5) Unanswerable: At least 1 question cannot be answered based on the visual information in the figure. The answer to this question can be "Cannot be determined", "Not enough information", "I don't know", etc.

3. **Provide Explanations**: In addition to a *concise answer* for each question, provide an explanation that details the reasoning steps to reach the answer. For the summary question, the explanation is a more detailed description of the figure.

4. **Response Format**: separate the question, answer, and explanation with a | character: question | answer | explanation. The question-answer pairs should be separated by double newlines (\n\n).
For example:
what's the difference between A and B? | 15 | A is 10 and B is 25, so the difference is 25-10=15

Which group has the highest value? A. Group-1 B. Group-2 C. Group-3 | B | Group-1 is 20, Group-2 is 25, and Group-3 is 15, so Group-2 has the highest value.

... ...

Do not include any additional text at the beginning or end of your response."""



GENERATE_CHART_QA_SHORT_ANSWER_PROMPT = """You are an expert in data analysis and good at asking questions about plots.
My persona is: "{persona}"
I want you to generate some question-answer pairs of a {figure_type} about {topic}, which I would ask.
Instead of showing the figure, I provide the data and the code that generates the figure.

<data>
{data}
</data>

<code>
{code}
</code>

Please come up with a list of *reasonable questions* that people will ask when they see the rendered figure. Here are the requirements:
1. **Question Style**: The questions must be natural and related to the figure, which can help interpret the data and understand the insights.
    (1) The questions vary in complexity. Some are easy to answer by just referring to the figure, and some are challenging and require multiple-step reasoning.
    (2) The questions should be answerable based on the *visual information* in the figure. Don't include any coding details in the questions since this type of information is not visible in the figure.

2. **Question Types**: All questions should be short-answer questions that are answerable based on the visual information in the figure.
    (1) **Basic questions** ask for basic elements in the figure such as visual appearance, title, axis, ticks, colors, number of data points, etc. Remeber to use **natural language** when referring to these elements, do not use coding terms (RGB, HEX, etc.).
    (2) **Compositional questions** contain at least two mathematical/logical operations like sum, difference, and average, e.g., "What's the median value of group A?". Generate questions using different numbers and types of operations.
    (3) **Comparison questions** ask about the relationship between two or more data points, e.g., "Which group has the highest value?". Generate questions using different comparison operators (greater than, less than, equal to, etc.) and data points.
    (4) **Chart-Type Specific questions** are unique to the figure type of {figure_type}.

3. **Provide Explanations**: 
    (1) In addition to a *concise answer* (as short as possible) for each question, provide a *step-by-step explanation* that details the reasoning steps to reach the answer.
    (2) For complicated questions, provide a **more detailed explanation** to break down the problem-solving process.

4. **Response Format**: Use | to separate the question, explanation, and concise answer for each example. 
    (1) Follow this format: question | explanation | concise answer, e.g., what's the average value of group A? | There are 5 data points in group A, sum them up is (34 + 45 + 23 + 56 + 12) = 170, so the average is 170/5 = 34 | 34
    (2) Separate the question-answer pairs by double newlines (\n\n). question1 | explanation1 | answer1\n\nquestion2 | explanation2 | answer2\n\n...
    (3) Do not provide too many questions, 5-10 questions are enough. Focus on the diversity and quality of the questions with a very detailed explanation for challenging questions.
    (4) The concise answer should be as short as possible and directly answer the question. All words in the response should be processed in natural language, no coding terms and remove any unnecessary characters.
    
Please follow the format strictly and do not include any additional text at the beginning or end of your response."""




GENERATE_CHART_CODE_MATPLOTLIB_PROMPT = """You are an expert in data analysis and good at writing code (Python `matplotlib`) to generate plots.
My persona is: "{persona}"
I have some data about {topic} which can be used to generate a {figure_type}.

Here is the data (CSV format, already loaded as a pd.DataFrame):
<data>
{data}
</data>

Please define a Python function (using `matplotlib`) called `generate_plot` that generates a {figure_type} using the data provided. Here are the requirements:
1. **Style Requirements**:
    (1) Try to be creative and change the arguments (e.g., font, color, marker, etc) to make the plot look unique. Set the style by using `plt.style.use('{style_name}')`.
    (2) Consider the **scale of data** to select the appropriate design scale (axis range, figure size, font/marker size, etc) to ensure the information in the plot is clear and easy to understand, with no text overlapping etc.

2. **Code Requirements**: create a function called `generate_plot` that generates the chart using `matplotlib`.
    (1) The data, which is loaded as a pd.DataFrame is provided as the first argument for the function. The function has no other arguments. You may need to adjust the data format or hard code some data to fit the `matplotlib` specification.
    (2) Remember to import necessary libraries (e.g., `import numpy as np`, `import matplotlib.pyplot as plt`) at the beginning of the script.
    (3) The `generate_plot` function should save the plot to a BytesIO and then return the plot as a PIL Image object. **Do not close the BytesIO object.**
    (4) Use `bbox_inches='tight'` argument in `savefig` or `plt.tight_layout()`, ensuring the plot is saved with all the elements (title, labels, etc) visible.
    (5) Only define the function and do not call it. Do not show the plot. Save the plot with appropriate resolution. No need to show example usage.

3. **Output Requirements**:
    Put ```python at the beginning and ``` at the end of the script to separate the code from the text. This will help me easily extract the code.

Please don't answer with any additional text in the script, your whole response should be the Python code which can be directly executed."""




GENERATE_CHART_CODE_PLOTLY_PROMPT = """You are an expert in data analysis and good at writing code (Python `plotly`) to generate plots.
My persona is: "{persona}"
I have some data about {topic} which can be used to generate a {figure_type}.

Here is the data (CSV format, already loaded as a pd.DataFrame):
<data>
{data}
</data>

Please define a Python function (using `plotly`) called `generate_plot` that generates a {figure_type} using the data provided. Here are the requirements:
1. **Style Requirements**:
    (1) Try to be creative and change the default arguments (e.g., font, color, marker, etc) to make the plot style unique.
    (2) Consider the **scale of data** to select the appropriate design scale (axis range, figure size, font/marker size, etc) to ensure the information in the plot is clear and easy to understand, with no text overlapping etc.

2. **Code Requirements**: create a function called `generate_plot` that generates the chart using `plotly`. Do not use matplotlib or other libraries.
    (1) The data, which is loaded as a pd.DataFrame is provided as the first argument for the function. The function has no other arguments. You may need to adjust the data format to fit the `plotly` specification.
    (2) Remember to import necessary libraries (e.g., `import numpy as np`, `import plotly.express as px`) at the beginning of the script.
    (3) The `generate_plot` function should save the plot to a BytesIO and then return the plot as a PIL Image object. **Do not close the BytesIO object.**
    (4) Select appropriate margins and tight layout, ensuring the plot is saved with all the elements (title, labels, etc) visible. Rotate text if necessary to make sure no text overlapping.
    (5) Only define the function and do not call it. Do not show the plot. Save the plot with enough resolution to be clearly visible. No need to show example usage.

3. **Output Requirements**:
    Put ```python at the beginning and ``` at the end of the script to separate the code from the text. This will help me easily extract the code.

Please don't answer with any additional text in the script, your whole response should be the Python code which can be directly executed."""




GENERATE_CHART_CODE_VEGALITE_PROMPT = """You are an expert in data analysis and good at writing Vega-Lite JSON to generate plots.
My persona is: "{persona}"
I have some data about {topic} which can be used to generate a {figure_type}.

Here is the data (JSON format):
<data>
{data}
</data>

Please use Vega-Lite to define a {figure_type} in JSON format using the data provided. Here are the requirements:
1. **Style Requirements**:
    (1) Try to be creative and change the default arguments (e.g., font, color, marker, etc) to make the plot style unique.
    (2) Consider the **scale of data** to select the appropriate design scale (axis range, figure size, font/marker size, etc) to ensure the information in the plot is clear and easy to understand, with no text overlapping etc.
    (3) Don't use rare fonts or colors that may not be supported by all viewers.

2. **Code Requirements**: define the chart using Vega-Lite (JSON format).
    (1) You need to hardcode the provided data into the JSON to generate the document. Be careful with the syntax and formatting of the Vega-Lite JSON.
    (2) Select appropriate margins and tight layout, ensuring the plot is saved with all the elements (title, labels, etc) visible.
    (3) Don't add actual data numbers in the plot. This is very important. Because you cannot align the data with figure elements.
    (4) If the data contains the year information, format the year correctly in the x-axis.

3. **Output Requirements**:
    Put ```vegalite at the beginning and ``` at the end of the script to separate the code from the text. This will help me easily extract the JSON.

Please don't answer with any additional text in the script, your whole response should be the Vega-Lite JSON which can be directly executed."""




GENERATE_CHART_CODE_LATEX_PROMPT = """You are an expert in data analysis and good at writing LaTex code to generate plots.
My persona is: "{persona}"
I have some data about {topic} which can be used to generate a {figure_type}.

Here is the data (CSV format):
<data>
{data}
</data>

Please write a LaTeX script to generate a {figure_type} using the data provided. Here are the requirements:
Here are the requirements:
1. **Style Requirements**:
    (1) Try to be creative and change the default arguments (e.g., font, color, marker, etc) to make the plot style unique. You can use common LaTeX packages such as ones that help with tables (booktabs, etc.), figures (pgfplots, etc.), and drawings (tikz, etc.).
    (2) Consider the **scale of data** to select the appropriate design scale (axis range, figure size, font/marker size, etc) to ensure the information in the plot is clear and easy to understand, with no text overlapping, etc.
    (3) Select appropriate margins and tight layout, ensuring the plot is saved with all the elements (title, labels, etc) visible.

2. **Code Requirements**:
    (1) You need to hardcode the provided data into the LaTeX script to generate the plot. Be careful with the syntax and formatting of the LaTeX script.
    (2) Use `standalone` LaTeX document class to generate the chart and add some border margin (`[border=xxpt]`). **Do not add the page number.**

3. **Output Requirements**:
    Put ```latex at the beginning and ``` at the end of the script to separate the code from the text. This will help me easily extract the code.

Please don't answer with any additional text in the script. Your whole response should be the LaTeX code, which can be directly executed."""




GENERATE_CHART_CODE_HTML_PROMPT = """You are an expert web designer and are good at writing HTML to create charts.
My persona is: "{persona}"
I have some data about {topic} which can be used to generate a {figure_type}.

Here is the data (JSON format):
<data>
{data}
</data>

Please use HTML and CSS to generate a {figure_type} using the data provided. Here are the requirements:
1. **Style Requirements**: Feel free to use any CSS framework, libraries, JavaScript plugins, or other tools to create the chart.
    (1) Try to be creative and make the web page style, fonts, colors, and visual layout unique with CSS. Taking persona, topic, and figure type into consideration when designing the plot.
    (2) Consider the **scale of data** to select the appropriate design scale (axis range, figure size, font/marker size, etc) to ensure the information in the plot is clear and easy to understand, with no text overlapping etc.
    (3) **Important:** Choose appropriate range of axis depends on the scale of data to ensure the information in the plot is easy to parse.

2. **Code Requirements**: 
    (1) You need to hardcode the provided data into the HTML script to generate the document. Be careful with the syntax and formatting of the HTML script.
    (2) Put everything in one HTML file. Do not use external CSS or JavaScript files. Set higher width of the plot to ensure the plot is clearly visible.

3. **Output Requirements**:
    Put ```html at the beginning and ``` at the end of the script to separate the code from the text.

Please don't answer with any additional text in the script, your whole response should be the HTML code which can be directly executed."""




GENERATE_CHART_CODE_BOKEH_PROMPT = """You are an expert in data analysis and good at writing code (Python `bokeh`) to generate plots.
My persona is: "{persona}"
I have some data about {topic} which can be used to generate a {figure_type}.

Here is the data (CSV format, already loaded as a pd.DataFrame):
<data>
{data}
</data>

Please define a Python function (using `bokeh`) called `generate_plot` that generates a {figure_type} using the data provided. Here are the requirements:
1. **Style Requirements**:
    (1) Try to be creative and change the default arguments (e.g., font, color, marker, etc) to make the plot style unique.
    (2) Consider the **scale of data** to select the appropriate design scale (axis range, figure size, font/marker size, etc) to ensure the information in the plot is clear and easy to understand, with no text overlapping etc.

2. **Code Requirements**: create a function called `generate_plot` that generates the chart using `bokeh`. Do not use matplotlib or other libraries.
    (1) The data, which is loaded as a pd.DataFrame is provided as the first argument for the function. The function has no other arguments. You may need to adjust the data format to fit the `plotly` specification.
    (2) Remember to import necessary libraries at the beginning of the script.
    (3) The `generate_plot` function should save the plot to a BytesIO and then return the plot as a PIL Image object. **Do not close the BytesIO object.**
    (4) Select appropriate margins and tight layout, ensuring the plot is saved with all the elements (title, labels, etc) visible.
    (5) Only define the function and do not call it. Do not show the plot. Save the plot with enough resolution to be clearly visible. No need to show example usage.

3. **Output Requirements**:
    Put ```python at the beginning and ``` at the end of the script to separate the code from the text. This will help me easily extract the code.

Please don't answer with any additional text in the script, your whole response should be the Python code which can be directly executed."""




CHART_QUESTION_TYPES = [
    "Basic: Ask for basic visual elements in the figure such as title, axis, ticks, colors, number of data points, etc.",
    "Visual: Ask about the visual attributes such as color, height, and length of graphical marks (e.g., bars) in the chart.",
    "Compositional: contains two mathematical/logical operations like sum, diference, average, median, etc.",
    "Compositional: contains three mathematical/logical operations like sum, diference, average, median, etc.",
    "Compositional: contains four mathematical/logical operations like sum, diference, average, median, etc.",
    "Compositional: contains five or more mathematical/logical operations like sum, diference, average, median, etc.",
    "Comparison: Ask about the relationship between two or more data points.",
    "Comparison: Ask about the relationship between three or more groups.",
    "Comparison: Ask about the relationship between four or more groups.",
    "Comparison: Ask about the relationship among all groups.",
    "Comparison: compare the data points under specific conditions.",
    "Data Retrieval: Ask for specific data points or values in the figure.",
    "Data Retrieval: Ask for some extreme (highest/lowest) data points or values in the figure.",
    "Data Retrieval: Ask for some non-extreme (somewhere in the middle) data points or values in the figure.",
    "Data Retrieval: Ask for the data points or values that meet specific conditions.",
    "Data Retrieval: Ask for the critical points (e.g., turning points, intersection, changing trend, etc) in the figure.",
    "Reasoning: Ask for a question requiring multi steps of reasoning to answer.",
    "Reasoning: Ask for a question requiring two steps of reasoning to answer.",
    "Reasoning: Ask for a question requiring three steps of reasoning to answer.",
    "Reasoning: Ask for a question requiring four steps of reasoning to answer.",
    "Reasoning: Ask for a question requiring five steps of reasoning to answer.",
    "Structure: Ask about the structure of the data or the figure.",
    "Counting: count the number of data points or groups or elements in the figure.",
    "Counting: count the number of data points under specific conditions.",
    "Trend: Ask about the trend or pattern in the figure.",
    "Layout: Ask about the layout or arrangement of the elements (e.g., subplots) in the figure.",
    "Visual Reasoning: parsing the visual information in the figure, such as line intersection, comparison of areas/hights, etc.",
    "Fact Checking: given a statement, ask if it is true or false based on the figure.",
    "Yes/No: Ask a binary question that can be answered with yes or no.",
    "Yes/No: Ask a binary question that can be answered with yes or no with multi-step reasoning.",
    "Multiple Choice: Ask a multiple-choice question with hard negative options.",
    "Multiple Choice: Ask a multiple-choice question with 3-8 options.",
    "Multiple Choice: Ask a multiple-choice question with 4 options.",
    "Multiple Choice: Ask a multiple-choice question with more than 4 options.",
    "Long question: the question is long with detailed context and requires reasoning to answer.",
    "Short question: the question is short and can be answered directly by taking a glance at the figure.",
    "Annotations: Ask about annotations in the figure, such as text notes, highlights, or arrows pointing to specific parts.",
    "Interaction: Ask about interactions between different variables in the chart.",
]




GENERATE_CHART_QA_DIVERSE_PROMPT = """You are an expert in data analysis and good at asking questions about charts.
My persona is: "{persona}"
I want you to generate some question-answer pairs of a {figure_type} about {topic}, which I would ask.
Instead of showing the figure, I provide the data and the code that generates the figure.

<data>
{data}
</data>

<code>
{code}
</code>

Please come up with a list of *reasonable questions* that people will ask when they see the rendered figure. Here are the requirements:
1. **Question Style**: The questions must be natural and related to the figure, which can help interpret the data and understand the insights.
    (1) The questions vary in complexity. Some are easy to answer by just referring to the figure, and some are challenging and require multiple-step reasoning.
    (2) The questions should be answerable based on the *visual information* in the figure. Don't include any coding details in the questions since this type of information is not visible in the figure.

2. **Question Types**: All questions should be short-answer questions that are answerable based on the visual information in the figure.
    {selected_question_types}
    (4) Chart-Type Specific: Ask questions that are unique to the figure type - "{figure_type}".
    (5) Topic-Specific: Ask questions that are specific to the topic - "{topic}".

3. **Provide Explanations**: 
    (1) In addition to a *concise answer* (as short as possible) for each question, provide a *step-by-step explanation* that details the reasoning steps to reach the answer.
    (2) For complicated questions, provide a **more detailed explanation** to break down the problem-solving process.

4. **Response Format**: Use | to separate the question, explanation, and concise answer for each example. 
    (1) Follow this format: question | explanation | concise answer, e.g., what's the average value of group A? | There are 5 data points in group A, sum them up is (34 + 45 + 23 + 56 + 12) = 170, so the average is 170/5 = 34 | 34
    (2) Separate the question-answer pairs by double newlines (\n\n). question1 | explanation1 | answer1\n\nquestion2 | explanation2 | answer2\n\n...
    (3) Provide one question for each question type (5 in total). Focus on the diversity and quality of the questions with a very detailed explanation for challenging questions.
    (4) The concise answer should be **as short as possible** that directly answer the question. All words in the response should be processed in natural language, no coding terms and remove any unnecessary characters.
    
Please follow the format strictly and do not include any additional text at the beginning or end of your response."""