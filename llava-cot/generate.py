import json
import base64
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import re
from google import genai
from google.genai import types

# Initialize Gemini client
client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY"),
)

input_file = 'input.jsonl'
output_file = 'output.jsonl'
refusal_file = 'refusal.txt'

image_base_path = "./data/"

num_threads = 256
max_retries = 2

write_lock = threading.Lock()

processed_ids = set()
refusal_ids = set()

if os.path.exists(refusal_file):
    with open(refusal_file, 'r', encoding='utf-8') as f:
        for line in f:
            refusal_ids.add(line.strip())

stop_processing_event = threading.Event()

def process_line(line):
    if stop_processing_event.is_set():
        return
    try:
        data = json.loads(line)
        entry_id = data['id']
        if entry_id in processed_ids:
            print(f"Entry {entry_id} already processed, skipping.")
            return
        
        if entry_id in refusal_ids:
            print(f"Entry {entry_id} is in refusal list, skipping.")
            return
        
        if 'image' in data:
            image_path = data['image']
            image_full_path = os.path.join(image_base_path, image_path)
            if os.path.exists(image_full_path):
                # Read image file for later use with Gemini
                with open(image_full_path, 'rb') as img_file:
                    image_data = img_file.read()
            else:
                print(f"Image {image_full_path} not found, skipping entry {entry_id}.")
                return
        else:
            return

        conversations = data['conversations']
        hints = data.get('hints', [])
        for index, convo in enumerate(conversations):
            role = convo['from']
            value = convo['value']

            if role == 'gpt':
                retry_attempts = 0
                success = False
                while retry_attempts < max_retries:
                    # Get question from previous conversation
                    assert index > 0 and conversations[index - 1]['from'] == 'human'
                    conversations[index - 1]['value'] = conversations[index - 1]['value'].replace('<image>\n', '').replace('\n<image>', '')
                    question = conversations[index - 1]['value']
                    
                    standard_answer = value

                    # Create prompt text
                    prompt_text = (
                        "I have an image and a question that I want you to answer. I need you to strictly follow the format with four specific sections: SUMMARY, CAPTION, REASONING, and CONCLUSION. It is crucial that you adhere to this structure exactly as outlined and that the final answer in the CONCLUSION matches the standard correct answer precisely. To explain further: In SUMMARY, briefly explain what steps you'll take to solve the problem. In CAPTION, describe the contents of the image, specifically focusing on details relevant to the question. In REASONING, outline a step-by-step thought process you would use to solve the problem based on the image. In CONCLUSION, give the final answer in a direct format, and it must match the correct answer exactly. If it's a multiple choice question, the conclusion should only include the option without repeating what the option is. Here's how the format should look: <SUMMARY> [Summarize how you will approach the problem and explain the steps you will take to reach the answer.] </SUMMARY> <CAPTION> [Provide a detailed description of the image, particularly emphasizing the aspects related to the question.] </CAPTION> <REASONING> [Provide a chain-of-thought, logical explanation of the problem. This should outline step-by-step reasoning.] </REASONING> <CONCLUSION> [State the final answer in a clear and direct format. It must match the correct answer exactly.] </CONCLUSION> (Do not forget </CONCLUSION>!)"
                        "\n\nQuestion: " + question
                        "\n\nStandard answer: " + standard_answer
                    )

                    # Add hints if available
                    if hints:
                        added_hints = "".join([f"\nHint: {hint}" for hint in hints])
                        prompt_text += added_hints
                    
                    # Create content for Gemini API
                    contents = [
                        types.Content(
                            role="user",
                            parts=[
                                types.Part.from_image(image_data),
                                types.Part.from_text(prompt_text)
                            ],
                        ),
                    ]
                    
                    # Configure generation parameters
                    generate_content_config = types.GenerateContentConfig(
                        temperature=0.2,
                        top_p=0.95,
                        top_k=40,
                        max_output_tokens=8192,
                        response_mime_type="text/plain",
                    )

                    try:
                        # Call Gemini API
                        response = client.models.generate_content(
                            model="gemini-2.0-flash",
                            contents=contents,
                            config=generate_content_config,
                        )
                        
                        augmented_answer = response.text

                        pattern = r"<CONCLUSION>(.*?)</CONCLUSION>"
                        match = re.search(pattern, augmented_answer, re.DOTALL)
                        
                        if match:
                            augmented_answer_for_judge = match.group(1).strip()
                            
                            # Create judgment prompt
                            judge_text = (
                                "Evaluate whether the assistant's response is valid. Respond with 'valid' if the assistant's response is not a refusal and it aligns with the standard answer in meaning. Respond with 'invalid' if the response is a refusal or differs from the standard answer in a meaningful way. "
                                "A refusal means the assistant states it cannot recognize a specific person/object or refuses to answer the question. Do not consider a response to be a refusal just because it includes the word 'no' or other negative terms. "
                                f"Standard answer: {standard_answer} "
                                f"Assistant's response: {augmented_answer_for_judge}"
                            )
                            
                            # Create content for judgment
                            judge_contents = [
                                types.Content(
                                    role="user",
                                    parts=[types.Part.from_text(judge_text)],
                                ),
                            ]
                            
                            # Call Gemini API for judgment
                            judge_response = client.models.generate_content(
                                model="gemini-2.0-flash",
                                contents=judge_contents,
                                config=types.GenerateContentConfig(
                                    temperature=0,
                                    max_output_tokens=300,
                                )
                            )
                            
                            judgment = judge_response.text.strip().lower()
                            
                            if 'invalid' in judgment:
                                retry_attempts += 1
                                print(f"Assistant's response: {augmented_answer}")
                                print(f"Assistant's response is invalid. Retrying ({retry_attempts}/{max_retries})...")
                                time.sleep(1)
                            else:
                                conversations[index]['value'] = augmented_answer
                                print(f"Entry {entry_id}, message {index} processed successfully.")
                                success = True
                                break
                        else:
                            retry_attempts += 1
                            print(f"Assistant's response: {augmented_answer}")
                            print(f"Assistant's response is invalid. Retrying ({retry_attempts}/{max_retries})...")
                            time.sleep(1)

                    except Exception as e:
                        print(f"An error occurred while processing entry {entry_id}, message {index}: {e}")
                        print(f"Process terminating...")
                        stop_processing_event.set()
                        return

                if not success:
                    print(f"Entry {entry_id} failed after max retries, adding to refusal list.")
                    refusal_ids.add(entry_id)
                    with write_lock:
                        with open(refusal_file, 'a', encoding='utf-8') as f:
                            f.write(f"{entry_id}\n")

                    del conversations[index]
                    if index > 0 and conversations[index - 1]['from'] == 'human':
                        del conversations[index - 1]
                        index -= 1
                    continue

        if conversations:
            with write_lock:
                with open(output_file, 'a', encoding='utf-8') as outfile:
                    json.dump(data, outfile, ensure_ascii=False)
                    outfile.write('\n')

                processed_ids.add(entry_id)

    except json.JSONDecodeError:
        print("Invalid JSON format, skipping line.")
        return

if os.path.exists(output_file):
    with open(output_file, 'r', encoding='utf-8') as outfile:
        for line in outfile:
            try:
                existing_data = json.loads(line)
                processed_ids.add(existing_data['id'])
            except json.JSONDecodeError:
                continue

with open(input_file, 'r', encoding='utf-8') as infile:
    lines = infile.readlines()

with ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = [executor.submit(process_line, line) for line in lines]

    for future in as_completed(futures):
        future.result()  

print("Processing complete.")