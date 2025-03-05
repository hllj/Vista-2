NUM_TOPICS = 5




GENERATE_DOCUMENT_TOPICS_PROMPT = """You are an expert in document generation and have a broad knowledge of different topics.
My persona is: "{persona}"
I want you to generate {num_topics} topics for {figure_type} that I will be interested in or I may see during my daily life given my persona.

Here are the requirements:
1. Each topic is a high-level summary of the contents in {figure_type} with some design details, e.g., "the utility bill for the month of January 2022 with a detailed breakdown of charges".
2. The topics should be diverse to help me generate varied documents. Each topic should be unique and not overlap with others.
3. The topics are conditioned on the document type. Please ensure the topics you provided can be best visualized in "{figure_type}".
4. All topics must be in English, even if the persona is non-English.
5. List {num_topics} topics for "{persona}" and separate them with a | character, e.g., topic1 | topic2 | ...... | topic{num_topics}.
Do not include any additional text at the beginning or end of your response."""




GENERATE_DOCUMENT_DATA_JSON_PROMPT = """You are an expert in content creation and have broad knowledge about various topics.
My persona is: "{persona}"
I need some materials about "{topic}", which can be used to generate a {figure_type}. 
Here are the requirements:
1. The materials should be related to the topic and customized according to my persona. Its structure must be suitable for the {figure_type}.
2. The materials should be realistic, and the contents should be named using real-world entities. Do not use placeholder names like xxA, xxB, etc. Do not use template data like [Name], [Date], etc.
3. The materials should be diverse and contain information from different aspects of the topic to ensure the document is informative.
4. Do not provide too many materials. Just provide key pieces of information that are essential for a **one-page document.**
5. All materials must be in English, even if the persona is non-English.
Please provide the materials in JSON format without additional text at the beginning or end."""




GENERATE_DOCUMENT_DATA_JSON_FEW_PROMPT = """You are an expert in content creation and have broad knowledge about various topics.
My persona is: "{persona}"
I need some materials about "{topic}", which can be used to generate a {figure_type}. 
Here are the requirements:
1. The materials should be related to the topic and customized according to my persona. Its structure must be suitable for the {figure_type}.
2. The materials should be realistic, and the contents should be named using real-world entities. Do not use placeholder names like xxA, xxB, etc. Do not use template data like [Name], [Date], etc.
3. The materials should be diverse and contain information from different aspects of the topic to ensure the document is informative.
4. Do not provide too many materials. Just provide key pieces of information that are essential for a **one-page document.**
5. All materials must be in English, even if the persona is non-English.
Please provide the materials in JSON format without additional text at the beginning or end."""




GENERATE_DOCUMENT_QA_PROMPT = """You are an expert in document analysis and good at asking questions about documents.
My persona is: "{persona}"
I want you to generate some question-answer pairs of a {figure_type} about {topic}, which I would ask.
Instead of showing the document, I provide the materials and the code that generates the document.

Here is the data:
{data}

Here is the code that generates the document:
```
{code}
```

Please come up with a list of *reasonable questions* that people will ask when they see the rendered document. Here are the requirements:
1. **Question Style**: The questions must be natural and related to the document, which can help interpret the data and understand the insights.
    (1) The questions vary in complexity. Some are easy to answer by just referring to the document, and some are challenging and require multiple-step reasoning.
    (2) The questions should be answerable based on the *visual information* in the document. Don't include any coding details in the questions since this type of information is not visible in the document.

2. **Question Types**: Most questions are short-answer, but some can be multiple-choice, yes/no, or summary questions. You can use the following types:
    (1) Short-answer: At least 5 short-answer questions.
    (2) Multiple-choice: There should be at least two multiple-choice questions. The number of options can be 3, 4, 5, or more. The option labels can be different types: alphabet, Arabic numerals, or Roman numerals. The correct option should be different in each question.
    (3) Yes/No (True/False): At least 1 binary question.
    (4) Summary: At least 1 summary question that asks for describing the *entire document* or the *main idea* of the document.
    (5) Unanswerable: At least 1 question cannot be answered based on the visual information in the document. The answer to this question can be "Cannot be determined", "Not enough information", "I don't know", etc.

3. **Provide Explanations**: In addition to a *concise answer* for each question, provide an explanation that details the reasoning steps to reach the answer. For the summary question, the explanation is a more detailed description of the document.

4. **Response Format**: separate the question, answer, and explanation with a | character: question | answer | explanation. The question-answer pairs should be separated by double newlines (\n\n).
For example:
what is the total revenue? | $100,000 | The total revenue is the sum of all revenue sources in the document.

which product has the highest sales? A. Product A B. Product B C. Product C | B | Product A - $10,000, Product B - $15,000, Product C - $5,000. Product B has the highest sales.

... ...

Do not include any additional text at the beginning or end of your response."""




GENERATE_DOCUMENT_QA_SHORT_ANSWER_PROMPT = """You are an expert in data analysis and good at asking questions about documents.
My persona is: "{persona}"
I want you to generate some question-answer pairs of a {figure_type} about {topic}, which I would ask.
Instead of showing the document, I provide the data and the code that generates the document.

<data>
{data}
</data>

<code>
{code}
</code>

Please come up with a list of *reasonable questions* that people will ask when they see the rendered document. Here are the requirements:

1. **Question Types**: All questions should be short-answer questions that are answerable based on the visual information in the document. All questions can be answered with a single word, phrase, or number. (as short as possible)
    (1) **Information Retrieval questions** ask for specific information in the document, such as numbers, names, dates, titles, etc. The questions should cover different aspects (areas) of the document. This is the most common type of question.
    (2) **Reasoning questions** require reasoning over multiple information in the document. These questions should be more challenging and require a deeper understanding of the document.
    (3) **Document Type-specific questions** are questions that are specific and unique to this document type {figure_type}. These questions should be tailored to the content and structure of the document.

2. **Response Format**: Use | to separate the question, explanation, and concise answer for each example. 
    (1) Follow this format: question | explanation | concise answer, e.g., what is the total revenue? | The total revenue is the sum of all revenue sources in the document, which is $2000 + $3000 + $5000 = $10000. | $10000 
    (2) Separate the question-answer pairs by double newlines (\n\n). question1 | explanation1 | answer1\n\nquestion2 | explanation2 | answer2\n\n...
    (3) Do not provide too many questions, 5-10 questions are enough. Focus on the diversity and quality of the questions. Try to cover different aspects of the document.
    (4) The concise answer should be as short as possible and directly answer the question. The answer should be faithful and exactly the same as what you would expect to see in the document, don't rephrase it. All words in the answer should be processed in natural language, no coding terms/characters.
    
Please follow the format strictly and do not include any additional text at the beginning or end of your response."""




GENERATE_DOCUMENT_CODE_LATEX_PROMPT = """You are an expert in content creation and good at writing LaTeX code to generate documents.
My persona is: "{persona}"
I have some materials about {topic} which can be used to generate a {figure_type}.

Here are the materials (JSON format):
<data>
{data}
</data>

Please write a LaTeX script to generate a {figure_type} using the data provided. Here are the requirements:
1. **Style Requirements**:
    (1) Try to be creative and change the default arguments (e.g., font, color, border, shade, etc) to make the document style unique while taking topics and person into consideration.
    (2) Select the appropriate design, layout, and margin, ensuring the document is saved with all the elements visible, clear, and easy to understand, with no text overlapping, etc.
    (3) All content should be on **one page**. This is very important. Do not make the document too long or too short.

2. **Code Requirements**: Generate a **one-page** document using LaTeX.
    (1) You need to hardcode the provided data into the LaTeX script to generate the document. Be careful with the syntax and formatting of the LaTeX script.
    (2) **Do not use tikzpicture!** Use `pagecolor` instead if you want to change the background color of the page. Most of time, just use white background.
    (3) **Remove page number** from the document. Do not try to insert the example image/figure into the document (e.g., don't use `\includegraphics[]example-image`).

3. **Output Requirements**:
    Put ```latex at the beginning and ``` at the end of the script to separate the code from the text. This will help me easily extract the code.

Please don't answer with any additional text in the script. Your whole response should be the LaTeX code, which can be directly executed."""




GENERATE_DOCUMENT_CODE_HTML_PROMPT = """You are an expert web designer and are good at writing HTML to create documents.
My persona is: "{persona}"
I have some materials about {topic} which can be used to generate a {figure_type}.

Here are the materials (JSON format):
<data>
{data}
</data>

Please use HTML and CSS to generate a {figure_type} using the data provided. Here are the requirements:
1. **Style Requirements**: Feel free to use any CSS framework, libraries, JavaScript plugins, or other tools to create the document.
    (1) Try to be creative and make the web page style, fonts, colors, borders and visual layout unique with CSS. Taking persona, topic, and document type into consideration when designing the document.
    (2) Select the appropriate design scale (e.g., margins, page size, layout, etc) to ensure the information in the document is clear and easy to understand, with no text overlapping etc.
    (3) **Do not make the page too long or too sparse.** All contents should be in **one page**. This is very important.
    
2. **Code Requirements**: 
    (1) You need to hardcode the provided data into the HTML script to generate the document. Be careful with the syntax and formatting of the HTML script.
    (2) Put everything in one HTML file. Do not use external CSS or JavaScript files.

3. **Output Requirements**:
    Put ```html at the beginning and ``` at the end of the script to separate the code from the text.

Please don't answer with any additional text in the script, your whole response should be the HTML code which can be directly executed."""


GENERATE_DOCUMENT_CODE_DOCX_PROMPT = """You are an expert web designer and are good at using `.docx` to create documents.
My persona is: "{persona}"
I have some materials about {topic} which can be used to generate a {figure_type}.

Here are the materials (JSON format, already loaded as a dict):

{data}

Please define a Python function called `generate_document` that generates a {figure_type} using the data provided. Use `python-docx` to define the document and then return the `Document` object.
Here are the requirements:
1. **Style Requirements**:
    (1) Try to be creative and change the default arguments (e.g., font, color, border, page size, etc) to make the document style unique while taking topics and person into consideration.
    (2) Select appropriate design, layout and margin, ensuring the document is saved with all the elements visible, clear and easy to understand, with no text overlapping, etc.

2. **Code Requirements**: create a Python function called `generate_document` that defines the document using `python-docx` and returns the `Document` object.
    (1) The data, which is loaded as a dictionary is provided as the first argument for the function. The function has no other arguments.
    (2) Remember to import necessary libraries (e.g., `from docx import Document`) at the beginning of the script.
    (3) The `generate_document` function should return the `Document` object, e.g., `doc = Document() ... YOUR CODE ... return doc`. Be careful with the syntax and formatting of the `python-docx` script.
    (4) Only define the function and do not call it. Do not show the document. Save the document with enough resolution to be clearly visible. No need to show example usage.

3. **Output Requirements**:
    Put ```python at the beginning and ``` at the end of the script to separate the code from the text.

Please don't answer with any additional text in the script, your whole response should be the Python code which can be directly executed."""




GENERATE_DOCUMENT_POINT_PROMPT = """You are an expert web designer and are good at editing HTML to add points for localization usage.
Task description: You are given a {figure_type} about {topic} and need to generate pointing data for it.
You need to first create a point intent, as well as the name for those points, and then modify the HTML code to add the points that can be visualized in the rendered page.

Here is the orginal HTML code:
<code>
{code}
</code>

Here are the requirements:
1. **Point Intent**: 
    (1) The point intent is like a question or goal that people will ask about what to point out in this {figure_type}, e.g., "Point to the vegeterian dishes in the menu.", "Show me the total revenue in the table.", etc.
    (2) I want you to generate one intent for each of the following types:
        (i) **Simple Single Point**: A simple intent that requires pointing to a single element which can be recognized quickly.
        (ii) **Complex Single Point**: A complex intent that requires multi-hop reasoning or calculation.
        (iii) **Multiple Points**: An intent that requires pointing to multiple elements.
        (iv) **Specific Single Point**: An intent lead to a point that is unique to this type of document.
        (v) **{intent_type}**
    Please try your best to adhere to those types and create diverse intents that cover different aspects of the document. You can adjust the intent based on the content of the document.
    (3) Use the prefix of "{prefix}" to start the point intent. You can also create your own style if you think it's more suitable and natural.
    (4) Most intents will just point to **one element**, so you just need to change one line of code.

2. **Point Names**: Assign a name for all the points that satisfy the intent. The name should be short and descriptive, e.g., "Vegeterian Dishes", "Total Revenue", etc.

3. **Editing HTML**: I have figured out how to modify the HTML to add points, you just need to follow the instructions below:
    (1) I have added the styles (`point-container` and `location-point`) for the points in the original HTML code. You just need to use the provided CSS classes to add the points.
    (2) To add the points, you need to first identify the elements in the HTML that satisfy the intent. Then, apply the CSS classes to these elements to add points over them.
    (3) Usage: element --> <span class="point-container">element<span class="location-point"></span></span>, remember the element here can be any HTML element, no need to be a entire span, can be a single word.
    (4) The point should be precise and accurate, pointing to the exact element that satisfies the intent and should not be too coarse grained.
    (5) You don't need to provide the full HTML code, just the modified lines with the points added (You can have multiple points in one line). **Do not change any other parts of the HTML code**, 

The output format is:
<intent_1>
The goal or question about what to point out.
</intent_1>

<name_1>
A short name for the points.
</name_1>

<modified_lines_1>
The modified lines of the HTML code with the points added. Using the following format:
original line x --> modified line x
original line y --> modified line y
...
**Make sure the original line exactly matches the line in the HTML code.**
</modified_lines_1>

<intent_2>
...
</intent_2>

<name_2>
...
</name_2>

<modified_lines_2>
...
</modified_lines_2>

...

Please adhere to the output format and do not include any additional text in your response."""




INTENT_PREFIXES = [
    "Point to",
    "Point out",
    "Provide a point (points) for",
    "Show me",
    "Highlight",
    "Locate",
    "Generate a point (a list of points) for",
    "Find the",
    "Identify",
    "Mark",
    "If there is xxx, point to it",
    "If there is xxx, show me",
    "Help me find",
]




# POINT_INTENTS = [
#     "point to one simple element",
#     "point to one obvious element",
#     "point to one element that is easy to point to",
#     "point to one element that can be recognized quickly",
#     "point to a specific word",
#     "point to an element in the corner",
#     "point to an element on the left",
#     "point to an element on the right",
#     "point to an element in the middle",
#     "point to an element at the top",
#     "point to an element at the bottom",
#     "point to one element requiring attention",
#     "point to one element that requires reasoning",
#     "point to one element that requires math calculation",
#     "point to one element that requires comparison",
#     "point to one element that people may care about",
#     "point to one element that people will likely ask about",
#     "point to one element that people will likely ignore",
#     "point to one element that is interesting",
#     "point to something that is unique",
#     "point to one element that is important when reading this document",
#     "a question that leads to a simple point",
#     "a question that leads to two points",
#     "a question that leads to three points",
#     "a question that leads to four points",
#     "a question that leads to five points",
#     "a question that leads to six points",
#     "a question that leads to seven points",
#     "a question that leads to eight points",
#     "a question that leads to nine points",
#     "a question that leads to ten points",
#     "a question that leads to more than ten points",
#     "point to something in the same category",
#     "point to something based on a specific criteria",
#     "point to multiple elements",
#     "point to specific words",
#     "point to multiple elements in the corner",
#     "point to multiple elements on the left",
#     "point to multiple elements on the right",
#     "point to multiple elements in the middle",
#     "point to multiple elements at the top",
#     "point to multiple elements at the bottom",
#     "point to multiple elements requiring attention",
#     "point to multiple elements that requires reasoning",
#     "point to multiple elements that requires math calculation",
#     "point to multiple elements that requires comparison",
#     "point to multiple elements that are important when reading this document",
#     "point to elements that people may care about",
#     "point to elements that people may overlook",
#     "point to elements that are interesting",
#     "point to elements that people will likely ask about",
#     "point to elements that people will likely ignore",
#     "point to elements that are easy to miss",
#     "point to elements that are hard to see",
#     "point to one item that people may care about",
#     "point to one item that people may overlook",
#     "point to one item that are interesting",
#     "point to one item that people will likely ask about",
#     "point to one item that people will likely ignore",
#     "point to one item that are easy to miss",
#     "point to one item that are hard to see",
#     "creative point intent",
#     "a point that is unique to this type of document",
#     "a point that is specific to this document",
#     "a challenging point intent",
#     "a simple point intent",
#     "a straightforward point intent",
#     "a complex point intent",
#     "an element that is easy to point to",
#     "an element that is directly visible",
#     "an intent that has long and detailed requirements",
#     "an intent that has short and simple requirements",
#     "an intent that requires a lot of thinking",
#     "point to a button that user will likely click",
#     "point to a clickable element",
# ]

POINT_INTENTS = [
    "point to one simple icon",
    "point to one obvious icon",
    "point to one icon that is easy to point to",
    "point to one icon that can be recognized quickly",
    "point to a specific word",
    "point to an icon in the corner",
    "point to an icon on the left",
    "point to an icon on the right",
    "point to an icon in the middle",
    "point to an icon at the top",
    "point to an icon at the bottom",
    "point to one icon requiring attention",
    "point to one icon that requires reasoning",
    "point to one icon that requires math calculation",
    "point to one icon that requires comparison",
    "point to one icon that people may care about",
    "point to one icon that people will likely ask about",
    "point to one icon that people will likely ignore",
    "point to one icon that is interesting",
    "point to something that is unique",
    "point to one icon that is important when reading this document",
    "a question that leads to a simple point",
    "a question that leads to two points",
    "a question that leads to three points",
    "a question that leads to four points",
    "a question that leads to five points",
    "a question that leads to six points",
    "a question that leads to seven points",
    "a question that leads to eight points",
    "a question that leads to nine points",
    "a question that leads to ten points",
    "a question that leads to more than ten points",
    "point to something in the same category",
    "point to something based on a specific criteria",
    "point to multiple icons",
    "point to specific words",
    "point to multiple icons in the corner",
    "point to multiple icons on the left",
    "point to multiple icons on the right",
    "point to multiple icons in the middle",
    "point to multiple icons at the top",
    "point to multiple icons at the bottom",
    "point to multiple icons requiring attention",
    "point to multiple icons that requires reasoning",
    "point to multiple icons that requires math calculation",
    "point to multiple icons that requires comparison",
    "point to multiple icons that are important when reading this document",
    "point to icons that people may care about",
    "point to icons that people may overlook",
    "point to icons that are interesting",
    "point to icons that people will likely ask about",
    "point to icons that people will likely ignore",
    "point to icons that are easy to miss",
    "point to icons that are hard to see",
    "point to one icon that people may care about",
    "point to one icon that people may overlook",
    "point to one icon that are interesting",
    "point to one icon that people will likely ask about",
    "point to one icon that people will likely ignore",
    "point to one icon that are easy to miss",
    "point to one icon that are hard to see",
    "creative point intent",
    "a point that is unique to this type of document",
    "a point that is specific to this document",
    "a challenging point intent",
    "a simple point intent",
    "a straightforward point intent",
    "a complex point intent",
    "an icon that is easy to point to",
    "an icon that is directly visible",
    "an intent that has long and detailed requirements",
    "an intent that has short and simple requirements",
    "an intent that requires a lot of thinking",
    "point to a button that user will likely click",
    "point to a clickable icon",
]



GENERATE_SCREEN_POINT_PROMPT = """You are an expert web designer and are good at editing HTML to add points for localization usage.
Task description: You are given a {figure_type} about {topic} and need to generate pointing data for it. This will micic how user will click on the screen.
You need to first create a point intent, as well as the name for those points, and then modify the HTML code to add the points that can be visualized in the rendered page.

Here is the orginal HTML code:
<code>
{code}
</code>

Here are the requirements:
1. **Point Intent**: 
    (1) The point intent is like a question or goal that people will click on the screenshot of {figure_type}, e.g., "Point to the exit button.", "Show me the point to add this product to cart.", etc.
    (2) I want you to generate one intent for each of the following types:
        (i) **Simple Single Point**: A simple intent that requires pointing to a single element/icon which can be recognized quickly.
        (ii) **Complex Single Point**: A complex intent that requires multi-hop reasoning or calculation.
        (iii) **Single Clickable Visual Element**: An intent that requires pointing to a single visual element/icon, e.g., a button, a text field, etc.
        (iv) **Specific Single Point**: An intent lead to a point/icon that is unique to this type of screenshot.
        (v) **{intent_type}**
    Please try your best to adhere to those types and create diverse intents that cover different aspects of the screenshot. You can adjust the intent based on the content of the screenshot.
    (3) Use the prefix of "{prefix}" to start the point intent. You can also create your own style if you think it's more suitable and natural.
    (4) Most intents will just point to **one element**, so you just need to change one line of code.

2. **Point Names**: Assign a name for all the points that satisfy the intent. The name should be short and descriptive, e.g., "Exit Button", "Add to Cart", etc.

3. **Editing HTML**: I have figured out how to modify the HTML to add points, you just need to follow the instructions below:
    (1) I have added the styles (`point-container` and `location-point`) for the points in the original HTML code. You just need to use the provided CSS classes to add the points.
    (2) To add the points, you need to first identify the elements in the HTML that satisfy the intent. Then, apply the CSS classes to these elements to add points over them.
    (3) Usage: element --> <span class="point-container">element<span class="location-point"></span></span>, remember the element here can be any HTML element, no need to be a entire span, can be a single word.
    (4) The point should be precise and accurate, pointing to the exact element that satisfies the intent and should not be too coarse grained.
    (5) You don't need to provide the full HTML code, just the modified lines with the points added (You can have multiple points in one line). **Do not change any other parts of the HTML code**, 

The output format is:
<intent_1>
The goal or question about what to point out.
</intent_1>

<name_1>
A short name for the points.
</name_1>

<modified_lines_1>
The modified lines of the HTML code with the points added. Using the following format:
original line x --> modified line x
original line y --> modified line y
...
**Make sure the original line exactly matches the line in the HTML code.**
</modified_lines_1>

<intent_2>
...
</intent_2>

<name_2>
...
</name_2>

<modified_lines_2>
...
</modified_lines_2>

...

Please adhere to the output format and do not include any additional text in your response."""