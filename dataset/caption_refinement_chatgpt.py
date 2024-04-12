from typing import Dict

import os
import time
from tqdm import tqdm

import openai
openai.api_key = ""  # put your OpenAI token here

model_id2caption: Dict[str, Dict[str, str]] = None  # captions by BLIP
# {
#     "<model_id>": {
#         "description": "<blip_output>",
#         "category": "<ground_truth_category>"
#     },
#     ...
# }
model_ids = sorted(list(model_id2caption.keys()))

process_bar = tqdm(total=len(model_ids))

for model_id in model_ids:
    if os.path.exists(f"chatgpt/{model_id}.txt"):  # for resuming
        process_bar.update(1)
        continue

    caption = model_id2caption[model_id]
    description = caption["description"]
    category = caption["category"]

    process_bar.set_description(f"Processing {model_id}")

    prompt = "Given a description of furniture from a captioning model and its ground-truth category, " + \
        "please combine their information and generate a new short description in one line. " + \
        "The provided category must be the descriptive subject of the new description. " + \
        "The new description should be as short and concise as possible, encoded in ASCII. " + \
        "Do not describe the background and counting numbers. " + \
        "Do not describe size like 'small', 'large', etc. " + \
        "Do not include descrptions like 'a 3D model', 'a 3D image', 'a 3D printed', etc. " + \
        "Descriptions such as color, shape and meterial are very important, you should include them. " + \
        "If the old description is already good enough, you can just copy it. " + \
        "If the old description is meaningless, you can just only include the category.\n" + \
        "For example:\n" + \
        "Given 'a 3D image of a brown sofa with four wooden legs' and 'multi-seat sofa', " + \
        "you should return: a brown multi-seat sofa with wooden legs\n" + \
        "Given 'a pendant lamp with six hanging balls on the white background' and 'pendant lamp', " + \
        "you should return: a pendant lamp with hanging balls\n" + \
        "Given 'a black and brown chair with a floral pattern' and 'armchair', " + \
        "you should return: a black and brown floral armchair\n" + \
        "The above examples indicate that you should delete the redundant words in the old description, " + \
        "such as '3D image', 'four', 'six' and 'white background', " + \
        "and you must include the category name as the subject in the new description.\n"

    user_prompt = f"The old descriptions is '{description}', its category is '{category}', " + \
        "the new descriptions should be: "

    while True:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": prompt}, {"role": "user", "content": user_prompt}],
                temperature=0.2,
                top_p=0.1
            )
            break
        except Exception as e:
            print(f"OpenAI error: {str(e)}")
            if "Max retries exceeded with url" in str(e):
                time.sleep(5)

    reply = response["choices"][0]["message"]["content"]

    with open(f"chatgpt/{model_id}.txt", "w") as f:
        f.write(reply)

    process_bar.update(1)
