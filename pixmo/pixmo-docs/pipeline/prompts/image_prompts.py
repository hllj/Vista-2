NUM_TOPICS = 5




GENERATE_IMAGE_TOPICS_PROMPT = """You are an expert photographer and have a broad knowledge of different topics.
My persona is: "{persona}"
I want you to generate {num_topics} topics for {image_type} that represent what I may see in my daily life or I will be interested in.

Here are the requirements:
1. Each topic is a short description of a scene with some details that can be visualized in an image.
2. The topics should be diverse to help me generate varied images. Each topic should be unique and not overlap with others.
3. The topics are conditioned on the image type. Please ensure the topics you provided can be best visualized in "{image_type}".
4. List {num_topics} topics for "{persona}" and separate them with a | character, e.g., topic1 | topic2 | ...... | topic{num_topics}.
Do not include any additional text at the beginning or end of your response."""




GENERATE_IMAGE_DESCRIPTION_PROMPT = """You are an expert in prompting Generative AI image models like DALL-E and Midjourney and have broad knowledge about various topics.
My persona is: "{persona}"
I need an image description to prompt a model to generate an image of "{topic}", which can be used to generate a {image_type}. 
Here are the requirements:
1. The image description must be suitable for the {image_type}.
2. The contents are related to the topic and customized according to my persona.
3. The image description should be realistic and detailed such that the Generative AI image generator has enough description to generate a very specific image.
4. Please do not use more than five sentences to describe the image.
Please provide the image description without additional text at the beginning or end."""




GENERATE_IMAGE_QA_PROMPT = """You are an expert in prompting Generative AI image models and good at asking questions about images.
My persona is: "{persona}"
I want you to generate some question-answer pairs of a {image_type} about {topic}, which I would ask.
Instead of showing the image, I provide a detailed description of the image.

Here is the description:
<description>
{description}
</description>

Please come up with a list of *reasonable questions* that people will ask when they see the image. Here are the requirements:
1. **Question Style**: The questions must be natural and related to the image, which can help interpret the image and understand the image and the scene in the image.
    (1) The questions vary in complexity. Some are easy to answer by just by quickly glancing at the image, and some are challenging and require multiple-step reasoning.
    (2) The questions should be answerable based on the *visual information* in the image. Don't include any details specific to the description in the questions since this type of information is not visible in the image.

2. **Question Types**: Most questions are short-answer, but some can be multiple-choice, yes/no, or summary questions. You can use the following types:
    (1) Short-answer: At least 5 short-answer questions.
    (2) Multiple-choice: There should be at least two multiple-choice questions. The number of options can be 3, 4, 5, or more. The option labels can be different types: alphabet, Arabic numerals, or Roman numerals. The correct option should be different in each question.
    (3) Yes/No (True/False): At least 1 binary question.
    (4) Summary: At least 1 summary question that asks for describing the *entire image* or writing a caption for the image.
    (5) Unanswerable: At least 1 question cannot be answered based on the visual information in the image. The answer to this question can be "Cannot be determined", "Not enough information", "I don't know", etc.

3. **Provide Explanations**: In addition to a *concise answer* for each question, provide an explanation that details the reasoning steps to reach the answer. For the summary question, the explanation is a more detailed description of the image.

4. **Response Format**: separate the question, answer, and explanation with a | character: question | answer | explanation. The question-answer pairs should be separated by double newlines (\n\n).
For example:
Is the dog chasing the cat? | True | The image shows a black labrador chasing a orange cat in a lawn, therefore the answer is True.

How many dogs are there in this image? A. 2 B. 3 C. 4 | B | There are 3 dogs in the image, two of them are playing with a ball and one is sleeping on the grass.

... ...

Do not include any additional text at the beginning or end of your response."""



GENERATE_IMAGE_QA_SHORT_ANSWER_PROMPT = """You are an expert in prompting Generative AI image models and good at asking questions about images.
My persona is: "{persona}"
I want you to generate some question-answer pairs of a {image_type}, which I would ask.
Instead of showing the image, I provide a detailed description of the image.

Here is the description:
<description>
{description}
</description>

Please come up with a list of *reasonable questions* that people will ask when they see the image. Here are the requirements:
1. **Question Style**: The questions must be natural and related to the image, which can help interpret the image and understand the image and the scene in the image.
    (1) The questions vary in complexity. Some are easy to answer by just by quickly glancing at the image, and some are challenging and require multiple-step reasoning.
    (2) The questions should be answerable based on the *visual information* in the image. Don't include any details specific to the description in the questions since this type of information is not visible in the image.
    (3) Since the image is generated by AI, some of the details may not be captured in the image. When asking questions, consider the limitations of the AI-generated image and not ask questions that require fine-grained details.

2. **Question Types**: All questions should be short-answer questions that are answerable based on the description of the image.
    (1) You can use the question types in VQA (Visual Question Answering) tasks, such as "What", "Where", "When", "Who", "Why", "How many", "How much", "How", etc.
    (2) The questions should be diverse and cover different aspects of the image, such as objects, colors, locations, actions, etc.
    (3) Provide at most 5 short-answer questions. The question and answer must be faithful to the description of the image. You can generate fewer if you cannot come up with 5 questions from the description.

3. **Provide Explanations**: In addition to a *concise answer* (as short as possible) for each question, provide a *step-by-step explanation* that details the reasoning steps to reach the answer.

4. **Response Format**: Use | to separate the question, explanation, and concise answer for each example. 
    (1) Follow this format: question | explanation | concise answer, e.g., How many black dogs are there in the image sleeping? | There are 6 dogs in the image, 4 are sleeping, and 2 of those 4 are black, so there are 2 black dogs sleeping. | 2
    (2) Separate the question-answer pairs by double newlines (\n\n). question1 | explanation1 | answer1\n\nquestion2 | explanation2 | answer2\n\n...
    (3) The concise answer should be as short as possible and directly answer the question.

Please follow the format strictly and do not include any additional text at the beginning or end of your response."""