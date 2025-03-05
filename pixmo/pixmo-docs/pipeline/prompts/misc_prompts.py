NUM_TOPICS = 5




GENERATE_CHEMICAL_TOPICS_PROMPT = """You are an expert in chemistry and you are good at provide chemeical topics for different people.
My persona is: "{persona}"
I want you to generate {num_topics} topics for {figure_type} chemical structure that I will be interested in or I may be exposed to during my daily life given my persona.

Here are the requirements:
1. Each topic is a high-level summary of a {figure_type} chemicals, e.g., "The chemical structure of a aspirin molecule".
2. The topics should be diverse to help me generate varied chemicals. Each topic should be unique and not overlap with others.
3. All topics must be in English, even if the persona is non-English.
4. List {num_topics} topics for "{persona}" and separate them with a | character, e.g., topic1 | topic2 | ...... | topic{num_topics}.
Each topic must can be visualized as a chemical structure. Do not include any additional text at the beginning or end of your response."""




GENERATE_CHEMICAL_DATA_PROMPT = """You are an expert in chemistry and know the structure of different {figure_type} chemicals.
My persona is: "{persona}" and I'd like to know the structure of "{topic}".
Could you provide **one** SMILES representation of the chemical structure of "{topic}" for me?
Please provide the representation in the SMILES format without any additional text at the beginning or end of your response.
Put ```SMILES at the beginning and ``` at the end of the SMILES representation to separate the code from the text. This will help me easily extract the SMILES representation."""




GENERATE_CHEMICAL_CODE_RDKIT_PROMPT = """You are an expert in `rdkit` and you are good at generating chemical structures.
My persona is: "{persona}"
I have a SMILES representation of a {figure_type} "{topic}" and I want you to visualize its chemical structure using `rdkit`.

Here is the SMILES representation:
<SMILES>
{data}
</SMILES>

Please define a Python function (using `rdkit`) called `generate_chemical` that generates a {figure_type} using the SMILES provided. Here are the requirements:
1. The SMILES representation, which is loaded as a string is provided as the first argument for the function. The function has no other arguments.
2. Remember to import necessary libraries (e.g., `from rdkit import Chem`, etc) at the beginning of the script.
3. The `generate_chemical` function should save the chemical structure to a BytesIO and then return the chemical structure as a PIL Image object. **Do not close the BytesIO object.**
4. Only define the function and do not call it. Do not show the chemical structure. Save the chemical structure with appropriate resolution. No need to show example usage.
5. Put ```python at the beginning and ``` at the end of the script to separate the code from the text. This will help me easily extract the code.

Please don't answer with any additional text in the script, your whole response should be the Python code which can be directly executed."""




GENERATE_CHEMICAL_QA_PROMPT = """You are an expert in chemistry and good at asking questions about chemicals.
My persona is: "{persona}"
I want you to generate some question-answer pairs of a "{figure_type}" about "{topic}", which I would ask.
Instead of showing the chemicals, I provide the SMILES representation:

<SMILES>
{code}
</SMILES>

Please come up with a list of *reasonable questions* that people will ask when they see the rendered chemical structure. Here are the requirements:
1. **Question Types**: All questions should be short-answer questions that are answerable based on the visual information in the chemical. All questions can be answered with a single word, phrase, or number. (as short as possible)
    (1) **Descriptive questions** ask for the basic information in the chemical, such as the number of atoms, bonds, the name of the chemical, basic properties, etc.
    (2) **Reasoning questions** require reasoning over multiple information in the chemical. These questions should be more challenging and require a deeper understanding of the chemical. Usually, you need to iterate through the flow of the chemical and combine multiple pieces of information to answer these questions.
    (3) **Chemical Type-specific questions** are questions that are specific and unique to this chemical type {figure_type}. These questions should be tailored to the content and structure of the chemical.

2. **Provide Explanations**: 
    (1) In addition to a *concise answer* (as short as possible) for each question, provide a *step-by-step explanation* that details the reasoning steps to reach the answer.
    (2) For complex reasoning questions, the explanation should be more detailed and include all the necessary steps.

3. **Response Format**: Use | to separate the question, explanation, and concise answer for each example.
    (1) Follow this format: question | explanation | concise answer, e.g., which part of this chemical causes it to have bitter taste? | This chemical has the structure of alkaloid (see the nitrogen atom), which causes the bitter taste. | alkaloid.
    (2) Separate the question-answer pairs by double newlines (\n\n). question1 | explanation1 | answer1\n\nquestion2 | explanation2 | answer2\n\n...
    (3) Do not provide too many questions, 5-7 questions are enough. Focus on the diversity and quality of the questions with a very detailed explanation for challenging questions.
    (4) The concise answer should be as short as possible and directly answer the question. The answer should be faithful and exactly the same as what you would expect to see in the chemical, don't rephrase it. All words in the answer should be processed in natural language, no coding terms/characters.

Please follow the format strictly and do not include any additional text at the beginning or end of your response."""




GENERATE_MUSIC_TOPICS_PROMPT = """You are an expert musician and you are good at providing music topics for different people.
My persona is: "{persona}"
I want you to generate {num_topics} topics for {figure_type} muisc that I will be interested in or I may listen to during my daily life given my persona.

Here are the requirements:
1. Each topic is a high-level summary of a {figure_type} music, e.g., "80s country music about the life of a cowboy living in mississippi".
2. The topics should be diverse to help me generate varied music. Each topic should be unique and not overlap with others.
3. All topics must be in English, even if the persona is non-English.
4. List {num_topics} topics for "{persona}" and separate them with a | character, e.g., topic1 | topic2 | ...... | topic{num_topics}.
Do not include any additional text at the beginning or end of your response."""




GENERATE_MUSIC_DATA_PROMPT = """You are an expert in music composition and good at generating sheet music for {figure_type} music.
My persona is: "{persona}" and I'd like you to come up with some ideas for generating sheet music for "{topic}", which will then be rendered using lilypond.
Provide the necessary information for generating the sheet music, such as melody, harmony, rhythm, instrumentation, and any other relevant details.
The materials are for **one song** only, so you don't need to provide a lot of information.
Please provide the materials in the JSON format without any additional text at the beginning or end of your response."""




GENERATE_MUSIC_CODE_LILYPOND_PROMPT = """You are an expert in music composition and good at using `lilypond` to generate sheet music.
My persona is: "{persona}"
I have some materials of a {figure_type} about "{topic}" and I want you to use `lilypond` to render the sheet music.

Here is the data:
<data>
{data}
</data>

Here are the requirements:
1. You need to define the lilypound script to render the sheet music based on the data provided. You need to hard code the data in the script and be careful with the syntax of lilypond.
2. Try to cover all provided data in the script and do not modify the data. The sheet must be **one page** only, you can cut some parts if the music is too long.
3. Set appropriate margins (larger than 15) and spacing to make the sheet music look good. Remove all default headers and footers in the sheet music.
4. Put ```lilypond at the beginning and ``` at the end of the script to separate the code from the text. This will help me easily extract the code.

Please don't answer with any additional text in the script, your whole response should be the lilypond code which can be directly executed."""




GENERATE_MUSIC_QA_PROMPT = """You are an expert in music and good at asking questions about sheet music.
My persona is: "{persona}"
I want you to generate some question-answer pairs of a "{figure_type}" about "{topic}", which I would ask.
Instead of showing the sheet music, I provide the data and the code that generates the sheet music using `lilypond`.

<data>
{data}
</data>

<code>
{code}
</code>

Please come up with a list of *reasonable questions* that people will ask when they see the rendered sheet music. Here are the requirements:
1. **Question Types**: All questions should be short-answer questions that are answerable based on the visual information in the sheet music. All questions can be answered with a single word, phrase, or number. (as short as possible)
    (1) **Descriptive questions** ask for the basic information in the music sheet, such as the key signature, time signature, tempo, etc.
    (2) **Reasoning questions** require reasoning over multiple information in the music sheet. These questions should be more challenging and require a deeper understanding of the music.
    (3) **Music Type-specific questions** are questions that are specific and unique to this music type {figure_type}. These questions should be tailored to the content and structure of the music.

2. **Provide Explanations**: 
    (1) In addition to a *concise answer* (as short as possible) for each question, provide a *step-by-step explanation* that details the reasoning steps to reach the answer.
    (2) For complex reasoning questions, the explanation should be more detailed and include all the necessary steps.

3. **Response Format**: Use | to separate the question, explanation, and concise answer for each example.
    (1) Follow this format: question | explanation | concise answer, e.g., what is the key signature of this music? | The key signature is C major, which has no sharps or flats. | C major.
    (2) Separate the question-answer pairs by double newlines (\n\n). question1 | explanation1 | answer1\n\nquestion2 | explanation2 | answer2\n\n...
    (3) Do not provide too many questions, 5-7 questions are enough. Focus on the diversity and quality of the questions with a very detailed explanation for challenging questions.
    (4) The concise answer should be as short as possible and directly answer the question. The answer should be faithful and exactly the same as what you would expect to see in the music sheet, don't rephrase it. All words in the answer should be processed in natural language, no coding terms/characters.

Please follow the format strictly and do not include any additional text at the beginning or end of your response."""




GENERATE_CIRCUIT_TOPICS_PROMPT = """You are an expert electrical engineer and you are good at providing circuit topics for different people.
My persona is: "{persona}"
I want you to generate {num_topics} topics for {figure_type} circuit that I will be interested in or I may be exposed to during my daily life given my persona.

Here are the requirements:
1. Each topic is a high-level summary of a {figure_type} circuit, e.g., "A simple series circuit with a resistor and a capacitor for a charging system".
2. The topics should be diverse to help me generate varied circuits. Each topic should be unique and not overlap with others.
3. All topics must be in English, even if the persona is non-English.
4. List {num_topics} topics for "{persona}" and separate them with a | character, e.g., topic1 | topic2 | ...... | topic{num_topics}.
Do not include any additional text at the beginning or end of your response."""




GENERATE_CIRCUIT_DATA_PROMPT = """You are an expert in electrical engineering and good at designing different {figure_type} circuits.
My persona is: "{persona}" and I'd like you to provide some ideas for designing a "{topic}" circuit, which will then be rendered as diagram.
Provide the necessary information for designing the circuit, such as components, connections, voltage, current, and any other relevant details.
Please provide the materials in the JSON format without any additional text at the beginning or end of your response."""




GENERATE_CIRCUIT_CODE_LATEX_PROMPT = """You are an expert electrical engineer and you are good at generating circuit diagrams using `circuitikz` in LaTeX.
My persona is: "{persona}"
I want you to use `circuitikz` to design a {figure_type} circuit diagram about "{topic}".

Here are the requirements:
(1) Use the available functions in `circuitikz` to design the circuit diagram based on topic provided. Be careful with the syntax of LaTeX.
(2) Select appropriate margins and layout and **remove the page number**.
(3) Make sure there is **no overlapping text**, rotate the text or use smaller fontsize if necessary, and ensure the diagram is saved with all the elements visible.
(3) Put ```latex at the beginning and ``` at the end of the script to separate the code from the text. This will help me easily extract the code.
Please don't answer with any additional text in the script, your whole response should be the LaTeX code which can be directly executed."""




GENERATE_CIRCUIT_CODE_SCHEMDRAW_PROMPT = """You are an expert electrical engineer and you are good at generating circuit diagrams using `schemdraw`.
My persona is: "{persona}"
I want you to use `schemdraw` to design a {figure_type} circuit diagram about "{topic}".

Define a python function called `generate_circuit` that generates the circuit diagram using `schemdraw`. Here are the requirements:
(1) Use the **available basic elements** in `schemdraw` to design the circuit diagram based on topic provided. Be careful with the syntax of `schemdraw`.
(2) The output of the function should be a `schemdraw.Drawing` object that represents the circuit diagram. *Do not show the circuit diagram.** Do not call `.draw()`.
(3) Put ```python at the beginning and ``` at the end of the script to separate the code from the text. This will help me easily extract the code.
Example skeleton code:

```python
import schemdraw
# IMPORT OTHER LIBRARIES

def generate_circuit():
    d = schemdraw.Drawing()
    
    # YOUR DESIGN, DO NOT SHOW THE DRAWING DO NOT CALL .draw()
    
    return d
```

Please don't answer with any additional text in the script, your whole response should be the python code which can be directly executed."""



GENERATE_CIRCUIT_QA_PROMPT = """You are an expert electrical engineer and good at asking questions about circuit diagrams.
My persona is: "{persona}"
I want you to generate some question-answer pairs of a "{figure_type}" about "{topic}", which I would ask.
Instead of showing the circuit, I provide the data and the code that generates the circuit diagram using `schemdraw`.

<data>
{data}
</data>

<code>
{code}
</code>

Please come up with a list of *reasonable questions* that people will ask when they see the rendered circuit diagram. Here are the requirements:
1. **Question Types**: All questions should be short-answer questions that are answerable based on the visual information in the circuit. All questions can be answered with a single word, phrase, or number. (as short as possible)
    (1) **Descriptive questions** ask for the basic information in the circuit diagram, such as the components, connections, voltage, current, etc.
    (2) **Reasoning questions** require reasoning over multiple information in the circuit diagram. These questions should be more challenging and require a deeper understanding of the circuit.
    (3) **Circuit Type-specific questions** are questions that are specific and unique to this circuit type {figure_type}. These questions should be tailored to the content and structure of the circuit.

2. **Provide Explanations**: 
    (1) In addition to a *concise answer* (as short as possible) for each question, provide a *step-by-step explanation* that details the reasoning steps to reach the answer.
    (2) For complex reasoning questions, the explanation should be more detailed and include all the necessary steps.

3. **Response Format**: Use | to separate the question, explanation, and concise answer for each example.
    (1) Follow this format: question | explanation | concise answer, e.g., what is the voltage of this circuit? | The voltage is 5V, which is the potential difference between the two points. | 5V.
    (2) Separate the question-answer pairs by double newlines (\n\n). question1 | explanation1 | answer1\n\nquestion2 | explanation2 | answer2\n\n...
    (3) Do not provide too many questions, 5-7 questions are enough. Focus on the diversity and quality of the questions with a very detailed explanation for challenging questions.
    (4) The concise answer should be as short as possible and directly answer the question. The answer should be faithful and exactly the same as what you would expect to see in the circuit, don't rephrase it. All words in the answer should be processed in **natural language**, **no coding terms/characters**.

Please follow the format strictly and do not include any additional text at the beginning or end of your response."""




GENERATE_GRAPHIC_TOPICS_PROMPT = """You are an expert in graphic design and good at provide design ideas for different people.
My persona is: "{persona}"
I want you to generate {num_topics} topics for {figure_type} (vector graphics) that I will like or interest me.

Here are the requirements:
1. Each topic is a high-level summary of a {figure_type} graphic design, e.g., "An icon of a cat playing with a ball of yarn."
2. The topics should be diverse to help me generate varied vector graphics. Each topic should be unique and not overlap with others.
3. All topics must be in English, even if the persona is non-English.
4. List {num_topics} topics for "{persona}" and separate them with a | character, e.g., topic1 | topic2 | ...... | topic{num_topics}.
Each topic must can then be visualized as a vector graphic. Do not include any additional text at the beginning or end of your response."""




GENERATE_GRAPHIC_DATA_PROMPT = """You are an expert graphic designer and good at creating vector graphics for {figure_type}.
My persona is: "{persona}". I need you to provide some design ideas for "{topic}", which will then be rendered as a vector graphic.
Please provide the necessary information for designing the graphic, such as colors, shapes, sizes, and any other relevant details.
Please provide the materials in the JSON format without any additional text at the beginning or end of your response."""




GENERATE_GRAPHIC_CODE_SVG_PROMPT = """You are an expert graphic designer and good at writing SVG code for vector graphics.
My persona is: "{persona}". I want you to use SVG to design a {figure_type} graphic about "{topic}".

Here are the requirements:
(1) Customize the SVG code to represent the "{topic}". Choose the appropriate layout, margins, colors, and shapes and ensure that the SVG code is valid and can be rendered correctly. 
(2) The SVG should have enough size and resolution to be clearly visible when rendered.
(2) Put ```svg at the beginning and ``` at the end of the script to separate the code from the text. This will help me easily extract the code.

Please don't answer with any additional text in the script. Your whole response should be the SVG code, which can be directly executed."""



GENERATE_GRAPHIC_QA_PROMPT = """"""