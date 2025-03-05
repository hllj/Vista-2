NUM_TOPICS = 5




GENERATE_SCREEN_TOPICS_PROMPT = """You are an expert in UI/UX and have a broad knowledge of different topics.
My persona is: "{persona}"
I want you to generate {num_topics} topics for {figure_type} that I will be interested in or I may see during my daily life given my persona.

Here are the requirements:
1. Each topic is a high-level summary of the contents in {figure_type} with some design details, e.g., "a screenshot of an exercise tracking app with minimalist design".
2. The topics should be diverse to help me generate varied screenshots. Each topic should be unique and not overlap with others.
3. The topics are conditioned on the screenshot type. Please ensure the topics you provided can be best visualized in "{figure_type}".
4. All topics must be in English, even if the persona is non-English.
5. List {num_topics} topics for "{persona}" and separate them with a | character, e.g., topic1 | topic2 | ...... | topic{num_topics}.
Do not include any additional text at the beginning or end of your response."""




GENERATE_SCREEN_DATA_JSON_FEW_PROMPT = """You are an expert in UI/UX design and have broad knowledge about various topics.
My persona is: "{persona}"
I need some materials about "{topic}", which can be used to generate a screenshot of {figure_type}. 
Here are the requirements:
1. The materials should be related to the topic and customized according to my persona. Its structure must be suitable for the {figure_type}.
2. Make sure the materials have clickable elements, such as buttons, links, icons, or images, to make the screenshot interactive.
3. The materials should be realistic, and the contents should be named using real-world entities. Do not use placeholder names like xxA, xxB, etc. Do not use template data like [Name], [Date], etc.
4. All materials must be in English, even if the persona is non-English.
Please provide the materials in JSON format without additional text at the beginning or end."""




GENERATE_SCREEN_QA_PROMPT = """You are an expert in screenshot analysis and good at asking questions about screenshots.
My persona is: "{persona}"
I want you to generate some question-answer pairs of a {figure_type} about {topic}, which I would ask.
Instead of showing the screenshot, I provide the materials and the code that generates the screenshot.

Here is the data:
{data}

Here is the code that generates the screenshot:
```
{code}
```

Please come up with a list of *reasonable questions* that people will ask when they see the rendered screenshot. Here are the requirements:
1. **Question Style**: The questions must be natural and related to the screenshot, which can help interpret the data and understand the insights.
    (1) The questions vary in complexity. Some are easy to answer by just referring to the screenshot, and some are challenging and require multiple-step reasoning.
    (2) The questions should be answerable based on the *visual information* in the screenshot. Don't include any coding details in the questions since this type of information is not visible in the screenshot.

2. **Question Types**: Most questions are short-answer, but some can be multiple-choice, yes/no, or summary questions. You can use the following types:
    (1) Short-answer: At least 5 short-answer questions.
    (2) Multiple-choice: There should be at least two multiple-choice questions. The number of options can be 3, 4, 5, or more. The option labels can be different types: alphabet, Arabic numerals, or Roman numerals. The correct option should be different in each question.
    (3) Yes/No (True/False): At least 1 binary question.
    (4) Summary: At least 1 summary question that asks for describing the *entire screenshot* or the *main idea* of the screenshot.
    (5) Unanswerable: At least 1 question cannot be answered based on the visual information in the screenshot. The answer to this question can be "Cannot be determined", "Not enough information", "I don't know", etc.

3. **Provide Explanations**: In addition to a *concise answer* for each question, provide an explanation that details the reasoning steps to reach the answer. For the summary question, the explanation is a more detailed description of the screenshot.

4. **Response Format**: separate the question, answer, and explanation with a | character: question | answer | explanation. The question-answer pairs should be separated by double newlines (\n\n).
For example:
what is the total revenue? | $100,000 | The total revenue is the sum of all revenue sources in the screenshot.

which product has the highest sales? A. Product A B. Product B C. Product C | B | Product A - $10,000, Product B - $15,000, Product C - $5,000. Product B has the highest sales.

... ...

Do not include any additional text at the beginning or end of your response."""


GENERATE_SCREEN_CODE_HTML_PROMPT = """You are an expert web designer and are good at writing HTML to mimic the screenshot of different UI/UX designs.
My persona is: "{persona}"
I have some materials about {topic} which can be used to generate a {figure_type}.

Here are the materials (JSON format):
<data>
{data}
</data>

Please use HTML and CSS to generate a {figure_type} using the data provided. Here are the requirements:
1. **Style Requirements**: Feel free to use any CSS framework, libraries, JavaScript plugins, or other tools to create the screenshot.
    (1) Make sure the rendered HTML page looks like a real screenshot of the {figure_type} with the provided data and clickable elements.
    (2) Try to be creative and make the web page style, fonts, colors, borders and visual layout unique with CSS. Taking persona, topic, and screenshot type into consideration when designing the screenshot.
    (3) Select the appropriate design scale (e.g., margins, page size, layout, etc) to ensure the information in the screenshot is clear and easy to understand, with no text overlapping etc.
    (4) Make sure the width of the HTML is almost the same as the width of the screenshot, do not leave too much blank space on the left or right side of the screenshot (e.g., content="width=
    
2. **Code Requirements**: 
    (1) You need to hardcode the provided data into the HTML script to generate the screenshot. Be careful with the syntax and formatting of the HTML script.
    (2) Put everything in one HTML file. Do not use external CSS or JavaScript files.

3. **Output Requirements**:
    Put ```html at the beginning and ``` at the end of the script to separate the code from the text.

Please don't answer with any additional text in the script, your whole response should be the HTML code which can be directly executed."""