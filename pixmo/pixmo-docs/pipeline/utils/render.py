import os
import re
import random
import subprocess
import tempfile
from io import BytesIO
from shutil import rmtree
from PIL import ImageOps, Image
import vl_convert as vlc
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from pdf2image import convert_from_bytes
from playwright.sync_api import sync_playwright


def crop_whitespace(image):
    # Invert the image (assuming white background)
    inverted_image = ImageOps.invert(image.convert("RGB"))

    # Convert image to grayscale
    grayscale_image = inverted_image.convert("L")

    # Get bounding box of non-white pixels
    bbox = grayscale_image.getbbox()

    if bbox:
        # Expand bounding box with a 50px-100px buffer
        buffer_size = random.randint(50, 100)
        left = max(0, bbox[0] - buffer_size)
        upper = max(0, bbox[1] - buffer_size)
        right = min(image.width, bbox[2] + buffer_size)
        lower = min(image.height, bbox[3] + buffer_size)

        # Crop image using calculated bounding box
        cropped_image = image.crop((left, upper, right, lower))
        return cropped_image

    # If no bounding box found (all white image), return original image
    return image


def crop_background(image):
    # Get the background color from the top-left corner pixel
    bg_color = image.getpixel((0, 0))

    # Convert image to RGB mode if it's not already
    image = image.convert("RGB")

    # Create a mask where background color pixels are white, others are black
    mask = Image.new('L', image.size, 0)
    for x in range(image.width):
        for y in range(image.height):
            if image.getpixel((x, y)) == bg_color:
                mask.putpixel((x, y), 255)

    # Invert the mask
    mask = ImageOps.invert(mask)

    # Get bounding box of non-background pixels
    bbox = mask.getbbox()

    if bbox:
        # Expand bounding box with a 50px-100px buffer
        buffer_size = random.randint(50, 100)
        left = max(0, bbox[0] - buffer_size)
        upper = max(0, bbox[1] - buffer_size)
        right = min(image.width, bbox[2] + buffer_size)
        lower = min(image.height, bbox[3] + buffer_size)

        # Crop image using calculated bounding box
        cropped_image = image.crop((left, upper, right, lower))
        return cropped_image

    # If no bounding box found (all background color), return original image
    return image


def extract_html_width(html):
    pattern = r'max-width:\s*(\d+)px'
    match = re.search(pattern, html)

    if match: return int(match.group(1))
    else: return None


def render_html(html, full_page=True, random_width=True):
    html = html.replace("initial-scale=1.0", "initial-scale=2.0")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        
        height = 800
        # Extract the width of the HTML content
        width = extract_html_width(html)
        if width is None: width = 1200
        else:
            # if random_width is an integer, add a random value to the width
            if isinstance(random_width, int):
                width += random_width
            else:
                if random_width: 
                    width += random.randint(50, 150)  # Add some buffer to the width
                else:
                    width += 100  # Add a fixed buffer to the width

        # Set the resolution of the browser
        page = browser.new_page(viewport={"width": width, "height": height})
        
        page.set_content(html)
        page.wait_for_timeout(2000)  # wait for the content to render

        # Take a screenshot of the full page
        screenshot_bytes = page.screenshot(full_page=full_page)
        browser.close()

    image = Image.open(BytesIO(screenshot_bytes))
    return image

def extract_screen_width(html):
    pattern = r'width:\s*(\d+)px'
    match = re.search(pattern, html)

    if match: return int(match.group(1))
    else: return None

def render_screen(html, full_page=True, random_width=True):
    html = html.replace("initial-scale=1.0", "initial-scale=2.0")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        
        height = 800
        # Extract the width of the HTML content
        width = extract_screen_width(html)
        if width is None: width = 800
        else:
            # if random_width is an integer, add a random value to the width
            if isinstance(random_width, int):
                width += random_width
            else:
                if random_width: 
                    width += random.randint(0, 100)
        
        # Set the resolution of the browser
        page = browser.new_page(viewport={"width": width, "height": height})
        
        page.set_content(html)
        page.wait_for_timeout(2000)  # wait for the content to render

        # Take a screenshot of the full page
        screenshot_bytes = page.screenshot(full_page=full_page)
        browser.close()

    image = Image.open(BytesIO(screenshot_bytes))
    return image


def render_latex(latex_source):
    def compile_latex(compiler, latex_file, temp_dir):
        process = subprocess.Popen(
            [
                compiler,
                "-interaction=nonstopmode",
                "-output-directory",
                temp_dir,
                latex_file,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        stdout, stderr = process.communicate()
        return process.returncode, stdout

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()

    # Prepare paths
    latex_file = os.path.join(temp_dir, "temp.tex")
    pdf_file = os.path.join(temp_dir, "temp.pdf")

    # Write LaTeX content to a file
    with open(latex_file, "w", encoding="utf-8") as f:
        f.write(latex_source)

    # Try to compile with pdflatex, xelatex, and lualatex
    compilers = ["pdflatex", "xelatex", "lualatex"]
    for compiler in compilers:
        returncode, stdout = compile_latex(compiler, latex_file, temp_dir)
        if returncode == 0:
            break
    else:
        rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(f'Error encountered during LaTeX rendering with all compilers:\n{stdout.decode("utf-8")}')

    # Convert PDF bytes to images using pdf2image
    images = convert_from_bytes(open(pdf_file, "rb").read())

    # Cleanup temporary files (optional)
    os.remove(latex_file)
    os.remove(pdf_file)
    rmtree(temp_dir, ignore_errors=True)

    # Return the PIL image (assuming single-page PDF)
    if images:
        return crop_whitespace(images[0])  # Return the first page as a PIL image

    raise RuntimeError("PDF did not generate anything.")


def render_vegalite(vegalite_json):
    png_data = vlc.vegalite_to_png(vl_spec=vegalite_json, scale=random.choice([1.5, 2, 2.5, 3]))
    img_buffer = BytesIO(png_data)
    return Image.open(img_buffer)


def render_mermaid(mermaid_code):
    scale = random.choice([2, 3])

    temp_dir = tempfile.mkdtemp()
    output_mmd = os.path.join(temp_dir, "diagram.mmd")
    output_image = os.path.join(temp_dir, "diagram.png")

    # Save the Mermaid code to a temporary file
    with open(output_mmd, "w") as file:
        file.write(mermaid_code)

    # Call the Mermaid CLI to generate the diagram with the specified scale
    subprocess.run(["mmdc", "-i", output_mmd, "-o", output_image, "-s", str(scale)])

    # Read the generated image
    with open(output_image, "rb") as file:
        image_data = file.read()

    # Clean up the temporary files
    os.remove(output_mmd)
    os.remove(output_image)
    os.rmdir(temp_dir)

    # Load the image from the data
    img = Image.open(BytesIO(image_data))

    return img


def render_chemical(smiles, dpi=300):
    # Create a molecule object from SMILES
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None: return None

    # Add 2D coordinates to the molecule
    AllChem.Compute2DCoords(mol)

    # Set a base size based on the length of the molecule
    base_size_inches = 1 + (mol.GetNumAtoms() * 0.1)
    
    # Calculate pixel dimensions based on DPI
    img_size_pixels = int(base_size_inches * dpi)
    
    # Create a drawing object with Cairo
    drawer = Draw.MolDraw2DCairo(img_size_pixels, img_size_pixels)

    # Customize drawing options randomly
    opts = drawer.drawOptions()
    
    opts.addAtomIndices = random.choice([True, False])
    opts.addStereoAnnotation = random.choice([True, False])
    opts.atomLabelFontSize = int(14 * (img_size_pixels / 300))  # Scale font size with DPI
    opts.bondLineWidth = random.uniform(1.0, 2.0) * (img_size_pixels / 300)  # Scale line width with DPI
    opts.dblBondOffset = random.uniform(0.1, 0.4)
    opts.colorBonds = random.choice([True, False])
    
    # Random background color
    opts.backgroundColour = (random.random(), random.random(), random.random())

    # Randomly highlight atoms and bonds

    if random.random() < 0.5:
        num_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()
        
        highlight_atoms = random.sample(range(num_atoms), k=random.randint(0, num_atoms))
        highlight_bonds = random.sample(range(num_bonds), k=random.randint(0, num_bonds))

        # Generate random colors for highlighted atoms
        highlight_atom_colors = {i: (random.random(), random.random(), random.random()) 
                                for i in highlight_atoms}

    # Draw the molecule
    drawer.DrawMolecule(mol)

    # Finalize the drawing
    drawer.FinishDrawing()

    # Get the PNG data
    png_data = drawer.GetDrawingText()

    # Open the PNG data with Pillow
    img = Image.open(BytesIO(png_data))

    return crop_whitespace(img)


def render_music(lilypound_code):
    # make the temp directory under the current working directory TODO: Change this to a more permanent location
    temp_dir = "/Users/yueyang1/Documents/github/Sci-Fi-dev/pipeline/lilypond_music_pipeline/temp"

    # make sure the temp directory exists and is empty
    os.makedirs(temp_dir, exist_ok=True)

    # Clean up the intermediate files in the temp directory
    for file in os.listdir(temp_dir):
        file_path = os.path.join(temp_dir, file)
        os.remove(file_path)

    # Write LilyPond code to a file
    lilypond_file = os.path.join(temp_dir, "music.ly")
    with open(lilypond_file, "w") as f:
        f.write(lilypound_code)

    # Compile LilyPond code to create PNG
    try:
        subprocess.run(
            ["lilypond", "--png", "-dno-point-and-click", "-o", "music", lilypond_file],
            check=True,
            cwd=temp_dir,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error encountered during LilyPond rendering: {e}")
    
    # Read the generated PNG image
    png_file = os.path.join(temp_dir, "music.png")
    with open(png_file, "rb") as f:
        image_data = f.read()

    # Clean up the intermediate files in the temp directory
    for file in os.listdir(temp_dir):
        file_path = os.path.join(temp_dir, file)
        os.remove(file_path)
    
    # Load the image from the data
    img = Image.open(BytesIO(image_data))

    return crop_background(img)


def render_circuit(circuit):
    # circuit is a schemdraw.Drawing object
    temp_dir = tempfile.mkdtemp()
    output_image = os.path.join(temp_dir, "circuit.png")

    # randomly get a dpi value ranging from 100 to 300
    circuit.save(output_image, dpi=random.randint(100, 300))

    # Read the generated image
    with open(output_image, "rb") as file:
        image_data = file.read()

    # Clean up the temporary files
    os.remove(output_image)
    os.rmdir(temp_dir)

    # Load the image from the data
    img = Image.open(BytesIO(image_data))

    return img


def render_svg(svg_string):
    # Convert SVG string to PNG image with enough resolution for OCR
    import cairosvg
    png_data = cairosvg.svg2png(bytestring=svg_string.encode("utf-8"), dpi=random.randint(150, 300))
    img_buffer = BytesIO(png_data)
    return Image.open(img_buffer)


def render_asymptote(asymptote_code):
    temp_dir = tempfile.mkdtemp()
    output_asy = os.path.join(temp_dir, "diagram.asy")
    output_image = os.path.join(temp_dir, "diagram")

    # Save the Asymptote code to a temporary file
    with open(output_asy, "w") as file:
        file.write(asymptote_code)

    # Call the Asymptote CLI to generate the diagram TODO: CHANGE RESOLUTION
    resolution = random.randint(3, 6)
    subprocess.run([
        "asy",
        "-f", "png",
        "-render", str(resolution),
        "-o", output_image,
        output_asy
    ])

    # Read the generated image
    with open(output_image + ".png", "rb") as file:
        image_data = file.read()

    # Clean up the intermediate files in the temp directory
    for file in os.listdir(temp_dir):
        file_path = os.path.join(temp_dir, file)
        os.remove(file_path)

    os.rmdir(temp_dir)

    # Load the image from the data
    img = Image.open(BytesIO(image_data))

    return img