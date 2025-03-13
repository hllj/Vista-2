# Synthetic Data Generation Workflow

This implementation follows the Florence-2 data annotation workflow using Google's Gemini API for generating synthetic data. The workflow consists of three main phases as described in the original Florence-2 paper.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

2. Set up your Google Gemini API key:
   - Get an API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a `.env` file in the project root by copying `.env.example`
   - Add your API key to the `.env` file:
     ```
     GOOGLE_API_KEY=your_api_key_here
     ```

## Directory Structure

```
florence-2/
├── data/
│   └── images/     # Place your input images here
├── output/
│   ├── text/       # Text annotation results
│   ├── regions/    # Region annotation results
│   └── triplets/   # Text-phrase-region triplet results
├── config.py       # Configuration settings
├── specialists.py  # Specialist models implementation
├── filters.py      # Annotation filtering implementation
├── main.py         # Main workflow script
└── .env            # Environment variables (API keys)
```

## Usage

1. Place your images in the `data/images/` directory (supported formats: jpg, png)

2. Run the workflow:
```bash
python main.py
```

The script will process each image through the following phases:

### Phase 1: Initial Annotation with Specialists

- **Text Specialist**: Generates three types of text annotations:
  - Brief description: Concise caption focusing on main subjects
  - Detailed description: More comprehensive description with attributes and relationships
  - More detailed description: Thorough description with all visual elements and context

- **Region Specialist**: Detects objects and their locations in the image with:
  - Object names/categories
  - Bounding box coordinates
  - Confidence scores

- **Text-Phrase-Region Specialist**: Links text descriptions with detected regions by:
  - Extracting noun phrases from text descriptions
  - Matching phrases to detected regions
  - Assigning confidence scores to each match

### Phase 2: Data Filtering

- Filters text annotations based on:
  - Object count (removes texts with too many objects)
  - Action complexity (ensures sufficient action description)
  - Object complexity (ensures rich object descriptions)

- Filters region annotations based on:
  - Confidence scores (removes low-confidence detections)
  - Non-maximum suppression (removes overlapping boxes)

- Filters text-phrase-region triplets based on:
  - Phrase relevance (removes generic or irrelevant phrases)
  - Confidence thresholds (ensures high-quality matches)

### Output

The results are saved in JSON format in the respective output directories:

- `output/text/`: Contains text annotations for each image
- `output/regions/`: Contains region annotations with bounding boxes
- `output/triplets/`: Contains text-phrase-region triplets

## Configuration

You can modify the settings in `config.py` to adjust:

- Confidence thresholds
- Maximum objects per image
- Minimum complexity requirements
- Text generation parameters
- Blacklisted phrases

## Notes

- This implementation uses the Gemini API as the specialist model for all annotation types
- The filtering process follows the Florence-2 paper's guidelines for ensuring annotation quality
- The workflow is designed to be extensible - you can add more specialist models or filtering criteria as needed
- Error handling is implemented to ensure the workflow continues even if some steps fail 