import os
import nest_asyncio
from dotenv import load_dotenv
from llama_parse import LlamaParse

#Apply asyncio patch
nest_asyncio.apply()
load_dotenv()

print("Uploading and parsing document via LlamaParse Vision API...")

#Initialise the parser
#Setting result_type="markdown" is the critical step to preserve tabular gridlines
parser = LlamaParse(
    result_type = "markdown",
    verbose=True
)

#Parse the document
file_path = "../Source-Documents/PFMEA_Rev_01.pdf"
documents = parser.load_data(file_path)

#Save the markdown output to a file so we can inspect the preserved tables
output_file = "parsed_document_v1.md"
with open(output_file, "w", encoding="utf-8") as f:
    for doc in documents:
        f.write(doc.text + "\n\n")

print(f"Extraction complete! Structured Markdown saved to {output_file}")

