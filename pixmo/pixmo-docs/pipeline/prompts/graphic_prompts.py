NUM_TOPICS = 5




GENERATE_GRAPHIC_TOPICS_PROMPT = """You are an expert in create graphic-based questions for different people.
My persona is: "{persona}".
I want you to generate {num_topics} topics for {figure_type} question that I would like to know more about.
Here are the requirements:
1. Each topic is a high-level summary of a {figure_type} question with graphics as inputs, e.g., "A question about the area of a triangle with a given base and height" for "geometry questions".
2. The topics should be diverse to help me generate varied questions. Each topic should be unique and not overlap with others.
3. All topics must be in English, even if the persona is non-English.
3. Make sure the graphics in those topics are not too hard to be rendered in SVG or Asymptote.
4. List {num_topics} topics for "{persona}" and separate them with a | character, e.g., topic1 | topic2 | ...... | topic{num_topics}.
Each topic must can be visualized as a question with graphics. Do not include any additional text at the beginning or end of your response."""




GENERATE_GRAPHIC_DATA_ASYMPTOTE_PROMPT = """You are an expert in creating graphic-based questions for different people.
My persona is: "{persona}". I need you to provide some data for creating {figure_type} question about "{topic}" to test my knowledge.
Your response must include the graphic, question, a detailed explanation, and the correct answer.
Requirements:
1. The question should be relevant to the specified topic and customized to fit the given persona. Do not start with persona in the question.
2. Provide a detailed step-by-step explanation that leads to the short answer. Ensure the answer is concise (as short as possible) and accurately reflects the explanation.
3. Use Asymptote for diagrams or graphics (e.g., geometry). Be careful with the syntax and formatting of the Asymptote script. The graphic should be clear and easy to understand.
4. Use LaTeX to format all math notions in the question, explanation, and answer. Use double dollar signs `$$` for block math mode and single dollar signs `$` for inline math mode.
5. Return your response in the following format:

<asymptote>
YOUR GRAPHIC HERE USING ASYMPTOTE
</asymptote>

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




GENERATE_GRAPHIC_CODE_ASYMPTOTE_PROMPT = """"""




GENERATE_GRAPHIC_QA_PROMPT = """"""




GENERATE_GRAPHIC_DATA_SVG_PROMPT = """You are an expert in creating graphic-based questions for different people.
My persona is: "{persona}". I need you to provide some data for creating {figure_type} question about "{topic}" to test my knowledge.
Your response must include the graphic, question, a detailed explanation, and the correct answer.
Requirements:
1. The question should be relevant to the specified topic and customized to fit the given persona. Do not start with persona in the question.
2. Provide a detailed step-by-step explanation that leads to the short answer. Ensure the answer is concise (as short as possible) and accurately reflects the explanation.
3. Use SVG for diagrams or graphics (e.g., geometry). Be careful with the syntax and formatting of the SVG script. Make sure the SVG has enought margin/resolution to avoid cutting off the graphic. The graphic should be clear and easy to understand.
4. Use LaTeX to format all math notions in the question, explanation, and answer. Use double dollar signs `$$` for block math mode and single dollar signs `$` for inline math mode.
5. Return your response in the following format:

<graph>
YOUR GRAPHIC HERE USING SVG
</graph>

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





GENERATE_GRAPHIC_CODE_SVG_PROMPT = """"""




GENERATE_GRAPHIC_QA_PROMPT = """"""