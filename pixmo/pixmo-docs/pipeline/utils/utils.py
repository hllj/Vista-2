import json
import re
import copy
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageColor, ImageDraw
from io import StringIO
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from threadpoolctl import threadpool_limits

def contains_chinese(text):
    # Check if the text contains any Chinese characters
    return bool(re.search("[\u4e00-\u9FFF]", text))


# load the personas
PERSONAS = []
with open("./pipeline/persona.jsonl", "r") as f:
    for line in f:
        if not contains_chinese(line): PERSONAS.append(json.loads(line)["persona"])
        else: continue


def is_json_valid(json_str):
    try:
        obj = json.loads(json_str)
        if not isinstance(obj, dict) and not isinstance(obj, list):
            raise RuntimeError("No valid JSON.")
        return True
    except Exception:
        return False


def extract_json(input_string):
    # Using regex to identify JSON structures in the string
    json_match = re.search(r"(\{.*\}|\[.*\])", input_string, re.DOTALL)
    if json_match:
        extracted_json = json_match.group(0)
        try:
            # Convert the extracted JSON string into a Python dictionary or list of dictionaries
            json.loads(extracted_json)
            return extracted_json
        except json.JSONDecodeError:
            return ""
    else:
        return ""


def extract_code(input_string):
    # extract code from the input string
    code_match = re.search(r"```python(.*)```", input_string, re.DOTALL)
    if code_match:
        extracted_code = code_match.group(1).strip()
        return extracted_code
    else:
        code_match = re.search(r"```(.*)```", input_string, re.DOTALL)
        if code_match:
            extracted_code = code_match.group(1).strip()
            return extracted_code
        else:
            print("No valid code found")
            return None


def extract_schemdraw_code(input_string):
    # extract code from the input string
    code_match = re.search(r"```python(.*)```", input_string, re.DOTALL)
    if code_match:
        extracted_code = code_match.group(1).strip()
        #remove the .draw() function
        lines = extracted_code.split("\n")
        new_lines = [line for line in lines if ".draw(" not in line]
        return "\n".join(new_lines)
    else:
        code_match = re.search(r"```(.*)```", input_string, re.DOTALL)
        if code_match:
            extracted_code = code_match.group(1).strip()
            #remove the .draw() function
            lines = extracted_code.split("\n")
            new_lines = [line for line in lines if ".draw(" not in line]
            return "\n".join(new_lines)
        else:
            print("No valid code found")
            return None


def extract_latex(input_string):
    # extract code from the input string
    code_match = re.search(r"```latex(.*)```", input_string, re.DOTALL)
    if code_match:
        extracted_code = code_match.group(1).strip()
        return extracted_code
    else:
        code_match = re.search(r"```(.*)```", input_string, re.DOTALL)
        if code_match:
            extracted_code = code_match.group(1).strip()
            return extracted_code
        else:
            print("No valid code found")
            return None


def extract_svg(input_string):
    # extract code from the input string
    code_match = re.search(r"```svg(.*)```", input_string, re.DOTALL)
    if code_match:
        extracted_code = code_match.group(1).strip()
        return extracted_code
    else:
        code_match = re.search(r"```(.*)```", input_string, re.DOTALL)
        if code_match:
            extracted_code = code_match.group(1).strip()
            return extracted_code
        else:
            print("No valid code found")
            return None


def extract_html(input_string):
    # extract code from the input string
    code_match = re.search(r"```html(.*)```", input_string, re.DOTALL)
    if code_match:
        extracted_code = code_match.group(1).strip()
        return extracted_code
    else:
        code_match = re.search(r"```(.*)```", input_string, re.DOTALL)
        if code_match:
            extracted_code = code_match.group(1).strip()
            return extracted_code
        else:
            print("No valid code found")
            return None


def extract_mermaid(input_string):
    # extract code from the input string
    code_match = re.search(r"```mermaid(.*)```", input_string, re.DOTALL)
    if code_match:
        extracted_code = code_match.group(1).strip()
        return extracted_code
    else:
        code_match = re.search(r"```(.*)```", input_string, re.DOTALL)
        if code_match:
            extracted_code = code_match.group(1).strip()
            return extracted_code
        else:
            print("No valid code found")
            return None


def randomize_matplorlib_code(code):
    lines = code.split("\n")
    new_lines = []
    available_styles = plt.style.available

    styles_for_randomization = []
    for style in available_styles:
        if "dark_background" in style: continue
        else: styles_for_randomization.append(style)
    
    random.seed(len(code))
    random.shuffle(styles_for_randomization)

    for line in lines:
        if "style.use" in line:
            STYLE_IS_VALID = False
            for style in available_styles:
                if style in line:
                    STYLE_IS_VALID = True
                    break
            if not STYLE_IS_VALID:
                indent = len(line) - len(line.lstrip()) # check how many indents are there
                random_style = random.choice(styles_for_randomization)
                new_lines.append(f"{' ' * indent}plt.style.use('{random_style}')")
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    return "\n".join(new_lines)


def is_csv_valid(csv_str):
    try:   
        # return False if there are less than 2 lines
        lines = csv_str.split("\n")
        if len(lines) < 2: return False

        df = pd.read_csv(StringIO(csv_str))
        if not isinstance(df, pd.DataFrame):
            raise RuntimeError("No valid CSV.")
        return True
    except Exception:
        return False


def extract_csv(input_string):
    lines = [line.strip() for line in input_string.split("\n") if "," in line]
    csv = []
    num_columns = None

    for line in lines:
        if line.strip():  # Ignore empty lines
            columns = line.split(",")
            if num_columns is None:
                num_columns = len(columns)
            elif len(columns) != num_columns:
                continue
            csv.append(columns)

    return "\n".join([",".join(row) for row in csv])


def extract_SMILES(input_string):
    # extract SMILES from the input string
    code_match = re.search(r"```SMILES(.*)```", input_string, re.DOTALL)
    if code_match:
        extracted_code = code_match.group(1).strip()
        return extracted_code
    else:
        code_match = re.search(r"```(.*)```", input_string, re.DOTALL)
        if code_match:
            extracted_code = code_match.group(1).strip()
            return extracted_code
        else:
            print("No valid code found")
            return None


from rdkit import Chem
def is_SMILE_valid(smiles):
    """
    Check if the given SMILES representation is valid.
    
    Parameters:
    smiles (str): A string representing the SMILES structure.
    
    Returns:
    bool: True if the SMILES is valid, False otherwise.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False


def extract_math(input_string):
    math_example = {}
    # <question> </question> is the math question
    question_match = re.search(r"<question>(.*)</question>", input_string, re.DOTALL)
    if question_match: math_example["question"] = question_match.group(1).strip()

    # <explanation> </explanation> is the explanation
    explanation_match = re.search(r"<explanation>(.*)</explanation>", input_string, re.DOTALL)
    if explanation_match: math_example["explanation"] = explanation_match.group(1).strip()

    # <answer> </answer> is the answer
    answer_match = re.search(r"<answer>(.*)</answer>", input_string, re.DOTALL)
    if answer_match: math_example["answer"] = answer_match.group(1).strip()

    return math_example


def extract_math_asymptote(input_string):
    math_example = {}
    # <asymptote> </asymptote> is the graph
    graph_match = re.search(r"<asymptote>(.*)</asymptote>", input_string, re.DOTALL)
    if graph_match: math_example["graph"] = graph_match.group(1).strip()

    # <question> </question> is the math question
    question_match = re.search(r"<question>(.*)</question>", input_string, re.DOTALL)
    if question_match: math_example["question"] = question_match.group(1).strip()

    # <explanation> </explanation> is the explanation
    explanation_match = re.search(r"<explanation>(.*)</explanation>", input_string, re.DOTALL)
    if explanation_match: math_example["explanation"] = explanation_match.group(1).strip()

    # <answer> </answer> is the answer
    answer_match = re.search(r"<answer>(.*)</answer>", input_string, re.DOTALL)
    if answer_match: math_example["answer"] = answer_match.group(1).strip()

    return math_example


def extract_math_svg(input_string):
    math_example = {}
    # <graph> </graph> is the graph
    graph_match = re.search(r"<graph>(.*)</graph>", input_string, re.DOTALL)
    if graph_match: math_example["graph"] = graph_match.group(1).strip()

    # <question> </question> is the math question
    question_match = re.search(r"<question>(.*)</question>", input_string, re.DOTALL)
    if question_match: math_example["question"] = question_match.group(1).strip()

    # <explanation> </explanation> is the explanation
    explanation_match = re.search(r"<explanation>(.*)</explanation>", input_string, re.DOTALL)
    if explanation_match: math_example["explanation"] = explanation_match.group(1).strip()

    # <answer> </answer> is the answer
    answer_match = re.search(r"<answer>(.*)</answer>", input_string, re.DOTALL)
    if answer_match: math_example["answer"] = answer_match.group(1).strip()

    return math_example


def is_math_valid(math_example):
    return all(key in math_example for key in ["question", "explanation", "answer"])


def is_math_graphic_valid(math_example):
    return all(key in math_example for key in ["graph", "question", "explanation", "answer"])


def extract_lilypond(input_string):
    # extract code from the input string
    code_match = re.search(r"```lilypond(.*)```", input_string, re.DOTALL)
    if code_match:
        extracted_code = code_match.group(1).strip()
        return extracted_code
    else:
        code_match = re.search(r"```(.*)```", input_string, re.DOTALL)
        if code_match:
            extracted_code = code_match.group(1).strip()
            return extracted_code
        else:
            print("No valid code found")
            return None


def extract_point_html(input_string):
    point_examples = []

    for i in range(1, 10):
        if f"<intent_{i}>" not in input_string: continue
        point_example = {}

        # <intent> </intent> is the intent for pointing
        intent_match = re.search(f"<intent_{i}>(.*)</intent_{i}>", input_string, re.DOTALL)
        if intent_match: point_example["intent"] = intent_match.group(1).strip()

        # <name> </name> is the name of the points
        name_match = re.search(f"<name_{i}>(.*)</name_{i}>", input_string, re.DOTALL)
        if name_match: point_example["name"] = name_match.group(1).strip()

        # <modified_lines> </modified_lines> is the edited html
        lines_match = re.search(f"<modified_lines_{i}>(.*)</modified_lines_{i}>", input_string, re.DOTALL)

        try:
            if lines_match: 
                extracted_lines = lines_match.group(1).strip()
                point_example["modified_lines"] = []
                for line in extracted_lines.split("\n"):
                    if "-->" in line:
                        original, modified = line.split("-->")
                        point_example["modified_lines"].append((original.strip(), modified.strip()))
        except ValueError:
            continue # Skip if invalid formatting

        point_examples.append(point_example)

    return point_examples


def compute_white_px_ratio(image):
    # Compute the ratio of white pixels in the image
    white_px = 0
    total_px = 0
    for px in image.getdata():
        if px == (255, 255, 255):
            white_px += 1
        total_px += 1
    return white_px / total_px


def compute_major_px_ratio(image):
    # Compute the ratio of the most common pixel in the image
    px_count = {}
    for px in image.getdata():
        if px in px_count:
            px_count[px] += 1
        else:
            px_count[px] = 1
    
    max_px = max(px_count, key=px_count.get)
    return px_count[max_px] / len(image.getdata())


def process_image(image, max_size=(2560, 1440), major_px_threshold=0.95, aspect_ratio_threshold=5, filter_small=True):
    # Resize the image if it exceeds the maximum size
    if image.size[0] * image.size[1] > max_size[0] * max_size[1]:
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
    
    # Convert alpha channel to white background
    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        alpha = image.convert('RGBA').split()[-1]
        bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
        bg.paste(image, mask=alpha)
        image = bg.convert("RGB")
    else:
        image = image.convert("RGB")
    
    major_px_ratio = compute_major_px_ratio(image)
    if major_px_ratio > major_px_threshold: print("Warning: Image is monochromatic.", major_px_ratio); return None

    byte_array = image.tobytes()
    width, height = image.size

    aspect_ratio = max(width, height) / min(width, height)
    if aspect_ratio > aspect_ratio_threshold: print("Warning: Image aspect ratio is too high."); return None

    if min(width, height) < 128 and filter_small: print("Warning: Image is too small."); return None

    image_from_byte_array = Image.frombytes("RGB", (width, height), byte_array)
    
    return image_from_byte_array


def fix_latex_white_text(code):
    RGB_replacements = ["{128,128,128}",
                        "{41,128,185}",
                        "{0,102,204}",
                        "{255,140,0}",
                        "{34,139,34}",
                        "{102,178,255}",
                        "{144,238,144}",
                        "{240,128,128}",
                        "{221,160,221}",
                        "{255,160,122}"]
    
    HEX_replacements = ["808080",
                        "2980B9",
                        "0066CC",
                        "FF8C00",
                        "228B22",
                        "66B2FF",
                        "90EE90",
                        "F08080",
                        "DDA0DD",
                        "FFA07A"]

    fix = False

    if "{255,255,255}" in code:
        code = code.replace("{255,255,255}", random.choice(RGB_replacements))
        fix = True
    
    if "FFFFFF" in code:
        code = code.replace("FFFFFF", random.choice(HEX_replacements))
        fix = True

    return code, fix


def extract_points(image, point_color="#FF69B4"):
    # Bug with numpy causes process to freeze when using multiple threads for parallel processing due to buggy OpenBLAS implementation installed
    # https://github.com/numpy/numpy/issues/17752#issuecomment-1359079118
    with threadpool_limits(limits=1, user_api='blas'): 
        # input is a image with some points with the color of point_color
        # output is a list of (x, y) coordinates of the points
        # the idea is to do some clustering that pixels with the color of point_color that are connected will be considered as one point

        # convert the image to a numpy array
        image = image.convert("RGB")
        image_array = np.array(image)
        width, height = image_array.shape[:2]

        # turn the image to a binary image
        binary_image = np.all(image_array == np.array(ImageColor.getcolor(point_color, "RGB")), axis=-1)

        # get the coordinates of the points
        points = np.argwhere(binary_image)

        def check_connected(target_point, grounp_points):
            # check if any point in the group is connected to the target point
            for point in grounp_points:
                if np.linalg.norm(target_point - point) == 1:
                    return True

        # group the points that are connected
        points = list(map(tuple, points))
        groups = [[points[0]]]
        if len(points) > 1500:
            print(f"Warning: Extracted {len(points)} pixels for the point color from the image. Too many random pixels match the point color {point_color} in {image}, so this point for this image will be skipped.")
            raise RuntimeError("Skip the current point because too many random pixels match the point's color.")

        for point in points[1:]:
            connected = False
            for group in groups:
                if check_connected(np.array(point), np.array(group)):
                    group.append(point)
                    connected = True
                    break
            if not connected:
                groups.append([point])
        
        # get the center of each group
        centers = []
        for group in groups:
            group = np.array(group)
            center = np.mean(group, axis=0)
            centers.append(center.tolist())
        
        # normalized points will be (x,y) coordinates in the range of [0, 100], upper left corner is (0, 0)
        normalized_centers = [{"x": round(center[1] / height * 100, 1), "y": round(center[0] / width * 100, 1)} for center in centers]
        return centers, normalized_centers


def get_a_different_color(image):
    # Convert the image to RGB and then to a NumPy array
    image = image.convert("RGB")
    image_array = np.array(image)

    # Get all unique colors in the image in hex format
    unique_colors = set(
        "#{:02x}{:02x}{:02x}".format(r, g, b)
        for r, g, b in image_array.reshape(-1, 3)
    )

    # Load CSS4 colors from Matplotlib
    available_colors = list(mcolors.CSS4_COLORS.values())

    # Find the first color in available_colors not in unique_colors
    for color in available_colors:
        if color not in unique_colors:
            return color

    # Fallback if all predefined colors are in the image
    return "#000000" if "#000000" not in unique_colors else "#FFFFFF"


def find_unused_color(image):
    """
    Finds a color (hex code) that does not exist in the given PIL image.

    :param image: A PIL.Image object.
    :return: A hex color code (string) that is not present in the image.
    """
    # Convert the image to RGB and get all pixel colors
    image = image.convert("RGB")
    pixels = set(image.getdata())

    # Generate random colors until we find one that is not in the image
    while True:
        # Generate a random RGB color
        random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        if random_color not in pixels:
            # Convert to hex code and return
            return "#{:02x}{:02x}{:02x}".format(*random_color)


def insert_point_style_to_html(html_code, color="#FF69B4"):
    pointing_style = """
/* Styles for the points */
.point-container {
    position: relative;
    display: inline-block;
}

.location-point {
    position: absolute;
    width: 10px;
    height: 10px;
    background-color: COLOR;
    z-index: 1000;
    left: 50%; /* Changed from left: 0 to left: 50% */
    top: 50%;
    transform: translate(-50%, -50%); /* Changed to translate both X and Y */
}
"""
    pointing_style = pointing_style.replace("COLOR", color)
    # insert the pointing style to the html code
    html_code = html_code.replace("</style>", pointing_style + "\n</style>")

    return html_code


def modify_html(html_code, modified_lines):
    updated_html = []
    for line in html_code.split("\n"):
        replaced = False
        for original, modified in modified_lines:
            if original in line:
                updated_html.append(line.replace(original, modified))
                modified_lines = modified_lines[1:]
                replaced = True
                break
        if not replaced: updated_html.append(line)
        
    return "\n".join(updated_html)


def draw_points(image, list_of_points):
    # make a copy of the image
    image = image.copy()
    
    # Define colors for PIL in RGB format
    colors = ["#FF0000", "#008000", "#0000FF", "#FFFF00", "#FFC0CB", 
              "#FFA500", "#800080", "#00FFFF", "#A52A2A", "#00FF00"]
    
    draw = ImageDraw.Draw(image)
    
    for i, points in enumerate(list_of_points):
        color = colors[i % len(colors)]  # Loop through colors if there are more points than colors
        for point in points:
            y, x = point
            # Draw an ellipse representing a circle with a 10-pixel radius centered at (x, y)
            draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill=color)

    return image
            

if __name__ == "__main__":
    # Test the image processing function
    image = Image.open("1.png")
    image = image.convert("RGB")
    print(get_a_different_color(image))