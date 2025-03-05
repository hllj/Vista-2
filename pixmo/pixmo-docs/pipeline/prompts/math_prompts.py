NUM_TOPICS = 5




GENERATE_MATH_TOPICS_PROMPT = """You are an expert in math and you are good at provide math questions for different people.
My persona is: "{persona}"
I want you to generate {num_topics} topics for {figure_type} math questions that I will be interested in or I may want to learn given my persona.

Here are the requirements:
1. Each topic is a high-level summary of a {figure_type} math questions, e.g., "Calculate the distance between two points on a coordinate plane."
2. The topics should be diverse to help me generate varied math questions. Each topic should be unique and not overlap with others.
3. All topics must be in English, even if the persona is non-English.
4. List {num_topics} topics for "{persona}" and separate them with a | character, e.g., topic1 | topic2 | ...... | topic{num_topics}.
Each topic must can then be expanded as a math question. Do not include any additional text at the beginning or end of your response."""




GENERATE_MATH_DATA_PROMPT = """You are a math expert specializing in {figure_type} problems. 
My persona is: "{persona}". I need you to create a math question tailored to the topic "{topic}". 
Your response must include the question, a detailed explanation, and the correct answer. 

Requirements:
1. The question should be relevant to the specified topic and customized to fit the given persona. Do not start with persona in the question.
2. Provide a detailed step-by-step explanation that leads to the short answer. Ensure the answer is concise (as short as possible) and accurately reflects the explanation.
3. Use LaTeX to format all math equations in the question, explanation, and answer. Use double dollar signs `$$` for block math mode and single dollar signs `$` for inline math mode.
4. It is encouraged to have some visual demonstration in the question if possible, pure text is less preferred. For diagrams or figures (e.g., geometry), use Asymptote within [asy] ... [/asy] tags.
5. Return your response in the following format:
<question>
YOUR QUESTION HERE
</question>

<explanation>
YOUR EXPLANATION HERE (STEP-BY-STEP)
</explanation>

<answer>
YOUR ANSWER HERE (AS SHORT AS POSSIBLE)
</answer>

Please don't include any additional text in your response. Your entire response should be the question, explanation, and answer."""




GENERATE_MATH_CODE_LATEX_PROMPT = """You are good at writing LaTeX code to generate documents.
I would like you to render the following math question in LaTeX.

<question>
{question}
</question>

Here are the requirements:
1. Only render the provided question; do not add any additional text, and do not include the explanation or answer. Do not change the contents of the question, but you can adjust its formatting to satisfy the LaTeX requirements. It is encouraged to have some visual demonstration in the question if possible.
2. Try to be creative and change the default arguments (e.g., font, color, border, shade, etc) to make the document style unique. **Do not add the page number.**
3. Be careful with the syntax and formatting of the LaTeX script (pdflatex). Use the most common packages for math rendering; this will lead to better compatibility.
4. Use `standalone` LaTex document class to generate the table and add some border margin (`[border=xxpt]`). Do not try to insert the example image/figure into the document (e.g., don't use `\includegraphics[]example-image`).
5. Put ```latex at the beginning and ``` at the end of the script to separate the code from the text. This will help me easily extract the code.

Please don't answer with any additional text in the script. Your whole response should be the LaTeX code, which can be directly executed."""



GENERATE_MATH_QA_PROMPT = """"""