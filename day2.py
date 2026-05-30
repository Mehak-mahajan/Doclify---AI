import pdfplumber

# Step 1: Extract all text from PDF
all_text = ""

with pdfplumber.open("notes.pdf") as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            all_text += text + "\n"

print(f"Total characters extracted: {len(all_text)}")

# Step 2: Split into chunks of 500 characters
chunk_size = 500
chunks = []

for i in range(0, len(all_text), chunk_size):
    chunk = all_text[i:i+chunk_size]
    chunks.append(chunk)

print(f"Total chunks created: {len(chunks)}")
print()
print("--- First chunk ---")
print(chunks[0])
print()
print("--- Second chunk ---")
print(chunks[1])
