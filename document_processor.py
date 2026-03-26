import pdfplumber
import os

class DocumentProcessor:
    def __init__(self, chunk_size=3000):
        """
        Initializes the document processor.
        chunk_size: Approximate number of characters per chunk to stay within LLM token limits.
        """
        self.chunk_size = chunk_size

    def extract_text_from_pdf(self, file_path_or_bytes):
        """
        Extracts text from a local PDF path or file-like object using pdfplumber.
        Returns the full concatenated text.
        """
        full_text = []
        try:
            with pdfplumber.open(file_path_or_bytes) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        full_text.append(page_text)
                    
                    # Also extract tabular data if present
                    tables = page.extract_tables()
                    for table in tables:
                        for row in table:
                            # Clean row from Nones
                            cleaned_row = [str(cell) if cell is not None else "" for cell in row]
                            full_text.append(" | ".join(cleaned_row))
                            
            return "\n".join(full_text)
        except Exception as e:
            print(f"Error extracting PDF: {e}")
            return ""

    def chunk_text(self, text):
        """
        Very simple character-based chunking.
        Returns a list of string chunks.
        """
        if not text:
            return []
            
        chunks = []
        for i in range(0, len(text), self.chunk_size):
            chunks.append(text[i:i + self.chunk_size])
            
        return chunks

    def process_document(self, file_path_or_bytes, is_pdf=True):
        """
        Main entry point for documents. Returns a list of text chunks.
        """
        if is_pdf:
            text = self.extract_text_from_pdf(file_path_or_bytes)
        else:
            # Assuming it's already text if not PDF
            if isinstance(file_path_or_bytes, str) and os.path.exists(file_path_or_bytes):
                with open(file_path_or_bytes, "r", encoding="utf-8") as f:
                    text = f.read()
            elif isinstance(file_path_or_bytes, bytes):
                text = file_path_or_bytes.decode("utf-8")
            else:
                # E.g. Streamlit UploadedFile
                text = file_path_or_bytes.getvalue().decode("utf-8")
                
        return self.chunk_text(text)

# Basic test
if __name__ == "__main__":
    processor = DocumentProcessor()
    print("DocumentProcessor initialized.")
