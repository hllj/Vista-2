import os
from pathlib import Path
from typing import Dict, List, Any
import json
from tqdm import tqdm
import google.generativeai as genai
from dotenv import load_dotenv

from specialists import TextSpecialist, RegionSpecialist, TextPhraseRegionSpecialist
from filters import AnnotationFilter
from config import *

class SyntheticDataGenerator:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Configure Gemini API
        if GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
        else:
            raise ValueError("GOOGLE_API_KEY environment variable is not set. Please set it to your Gemini API key.")
        
        # Initialize specialists
        self.text_specialist = TextSpecialist()
        self.region_specialist = RegionSpecialist()
        self.triplet_specialist = TextPhraseRegionSpecialist()
        
        # Initialize filter
        self.filter = AnnotationFilter()
        
        # Create necessary directories
        self._setup_directories()
        
    def _setup_directories(self):
        """Create necessary directories if they don't exist."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        for subdir in ['text', 'regions', 'triplets']:
            (OUTPUT_DIR / subdir).mkdir(exist_ok=True)
            
    def process_image(self, image_path: str) -> Dict[str, Any]:
        """Process a single image through all annotation phases."""
        results = {}
        
        # Phase 1: Initial annotation with specialists
        print("Phase 1: Generating initial annotations...")
        
        # Generate text annotations
        try:
            text_results = self.text_specialist.process_image(image_path)
            results['text'] = text_results
            print(f"  ✓ Generated text annotations: {len(text_results)} types")
        except Exception as e:
            print(f"  ✗ Failed to generate text annotations: {str(e)}")
            results['text'] = {}
        
        # Generate region annotations
        try:
            region_results = self.region_specialist.process_image(image_path)
            results['regions'] = region_results
            print(f"  ✓ Generated region annotations: {len(region_results)} regions")
        except Exception as e:
            print(f"  ✗ Failed to generate region annotations: {str(e)}")
            results['regions'] = {}
        
        # Generate text-phrase-region triplets
        try:
            triplet_results = self.triplet_specialist.process_image(image_path)
            results['triplets'] = triplet_results
            print(f"  ✓ Generated triplets: {len(triplet_results)} triplets")
        except Exception as e:
            print(f"  ✗ Failed to generate triplets: {str(e)}")
            results['triplets'] = []
        
        # Phase 2: Filter and clean annotations
        print("Phase 2: Filtering and cleaning annotations...")
        
        # Filter text annotations
        try:
            filtered_text = self.filter.filter_text_annotations(results['text'])
            print(f"  ✓ Filtered text annotations: {len(filtered_text)}/{len(results['text'])} kept")
            results['text'] = filtered_text
        except Exception as e:
            print(f"  ✗ Failed to filter text annotations: {str(e)}")
        
        # Filter region annotations
        try:
            filtered_regions = self.filter.filter_region_annotations(results['regions'])
            print(f"  ✓ Filtered region annotations: {len(filtered_regions)}/{len(results['regions'])} kept")
            results['regions'] = filtered_regions
        except Exception as e:
            print(f"  ✗ Failed to filter region annotations: {str(e)}")
        
        # Filter triplets
        try:
            filtered_triplets = self.filter.filter_triplets(results['triplets'])
            print(f"  ✓ Filtered triplets: {len(filtered_triplets)}/{len(results['triplets'])} kept")
            results['triplets'] = filtered_triplets
        except Exception as e:
            print(f"  ✗ Failed to filter triplets: {str(e)}")
        
        return results
    
    def process_dataset(self, image_dir: str):
        """Process all images in a directory."""
        image_dir = Path(image_dir)
        image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        
        print(f"Found {len(image_files)} images to process")
        
        for image_file in tqdm(image_files, desc="Processing images"):
            try:
                print(f"\nProcessing {image_file.name}...")
                
                # Process image
                results = self.process_image(str(image_file))
                
                # Save results
                base_name = image_file.stem
                
                # Save text annotations
                with open(OUTPUT_DIR / 'text' / f"{base_name}.json", 'w') as f:
                    json.dump(results['text'], f, indent=2)
                    
                # Save region annotations
                with open(OUTPUT_DIR / 'regions' / f"{base_name}.json", 'w') as f:
                    json.dump(results['regions'], f, indent=2)
                    
                # Save triplets
                with open(OUTPUT_DIR / 'triplets' / f"{base_name}.json", 'w') as f:
                    json.dump(results['triplets'], f, indent=2)
                    
                print(f"Results saved for {image_file.name}")
                    
            except Exception as e:
                print(f"Error processing {image_file}: {str(e)}")
                continue

def main():
    # Set up the data generator
    try:
        generator = SyntheticDataGenerator()
        
        # Process images
        image_dir = DATA_DIR / "images"
        if not image_dir.exists():
            print(f"Please place your images in {image_dir}")
            image_dir.mkdir(exist_ok=True)
            return
            
        generator.process_dataset(str(image_dir))
        print("\nProcessing complete! Results are saved in the output directory.")
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Please make sure you have set the GOOGLE_API_KEY environment variable.")
        print("You can create a .env file in the project root with the following content:")
        print("GOOGLE_API_KEY=your_api_key_here")

if __name__ == "__main__":
    main() 