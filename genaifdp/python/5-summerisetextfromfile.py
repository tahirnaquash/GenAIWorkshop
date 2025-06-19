from xml.dom.minidom import Text
import requests
import PyPDF2

def download_file_from_google_drive(file_id, destination):
    URL = "https://drive.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    with open(destination, "wb") as f:
        f.write(response.content)

# Example usage:
file_id = "1ENwlS6fij2Lr-uXPKtgFbl7c2IJkpaWR"
destination = "Curriculum vitae1.pdf"
download_file_from_google_drive(file_id, destination)

# This function downloads a file from Google Drive using its file ID and saves it to the specified destination.
# It uses the requests library to handle the HTTP request and response.
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

pdf_text = extract_text_from_pdf(destination)

print("Extracted Text:\n", pdf_text[:500000])  # Print first 500 characters for brevity



from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")  # You can use any public summarization model

# If the text is too long, split it into chunks
max_chunk = 1000
chunks = [pdf_text[i:i+max_chunk] for i in range(0, len(pdf_text), max_chunk)]

summary = ""
for chunk in chunks:
    result = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
    summary += result[0]['summary_text'] + " "

print("Summary:\n", summary)

# Now, let's summarize the text in Kannada using a multilingual model
# Ensure you have the transformers library installed: pip install transformers
from transformers import pipeline

summarizer = pipeline(
    "summarization",
    model="csebuetnlp/mT5_multilingual_XLSum",
    tokenizer="csebuetnlp/mT5_multilingual_XLSum"
)

def summarize_in_kannada(text):
    prompt = f"kn: {text}"
    result = summarizer(prompt, max_length=130, min_length=30, do_sample=False)
    return result[0]['summary_text']

# Example usage:
summary = summarize_in_kannada("ನೀವು ಇಲ್ಲಿ ಕನ್ನಡ ಪಠ್ಯವನ್ನು ಹಾಕಬಹುದು.")
print(summary)