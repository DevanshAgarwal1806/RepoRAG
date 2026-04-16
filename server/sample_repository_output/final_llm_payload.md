USER QUERY: Which function handles cleaning text?

### CODEBASE CONTEXT ###

--- PRIMARY MATCH: DocumentProcessor.clean_text ---
File: server/sample_repository/text_processor.py
Code:
```
def clean_text(self, raw_text):
        """Cleans the text by removing punctuation and converting to lowercase."""
        # Calls another method within the same class
        no_punct = self.remove_punctuation(raw_text)
        
        # Calls string method (.lower)
        return no_punct.lower()
```

--- NEIGHBORING CONTEXT: __main__ ---
File: server/sample_repository/main.py
Code:
```
if __name__ == "__main__":
    sample_text = "This is a sample document to test the processing pipeline."
    run_pipeline(sample_text)
```

--- PRIMARY MATCH: run_pipeline ---
File: server/sample_repository/main.py
Code:
```
def run_pipeline(document_text):
    """Main execution function for the pipeline."""
    processor = DocumentProcessor(["the", "is", "and"])
    
    # Call to object method
    cleaned_doc = processor.clean_text(document_text)
    
    # Standalone function call
    score = calculate_bm25(5.0, 1.2)
    
    print(f"Processed document score: {score}")
    return score
```

--- PRIMARY MATCH: DocumentProcessor.remove_punctuation ---
File: server/sample_repository/text_processor.py
Code:
```
def remove_punctuation(self, text):
        # Calls string methods (.replace)
        return text.replace(".", "").replace(",", "")
```

--- NEIGHBORING CONTEXT: calculate_bm25 ---
File: server/sample_repository/math_utils.py
Code:
```
def calculate_bm25(tf, idf):
    """Calculates the BM25 score for a given term."""
    # Simplified for testing
    return tf * idf
```

