class DocumentProcessor:
    def __init__(self, stop_words):
        """Initializes the processor with stop words."""
        self.stop_words = stop_words

    def remove_punctuation(self, text):
        # Calls string methods (.replace)
        return text.replace(".", "").replace(",", "")

    def clean_text(self, raw_text):
        """Cleans the text by removing punctuation and converting to lowercase."""
        # Calls another method within the same class
        no_punct = self.remove_punctuation(raw_text)
        
        # Calls string method (.lower)
        return no_punct.lower()