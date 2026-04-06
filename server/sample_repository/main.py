from text_processor import DocumentProcessor
from math_utils import calculate_bm25

def run_pipeline(document_text):
    """Main execution function for the pipeline."""
    processor = DocumentProcessor(["the", "is", "and"])
    
    # Call to object method
    cleaned_doc = processor.clean_text(document_text)
    
    # Standalone function call
    score = calculate_bm25(5.0, 1.2)
    
    print(f"Processed document score: {score}")
    return score

if __name__ == "__main__":
    sample_text = "This is a sample document to test the processing pipeline."
    run_pipeline(sample_text)