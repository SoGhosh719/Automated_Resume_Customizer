import PyPDF2
import docx

def extract_text_from_pdf(file):
    file.seek(0)  # Ensure pointer is at start
    pdf = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def extract_text_from_docx(file):
    file.seek(0)  # Required for in-memory files
    doc = docx.Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def extract_resume_text(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    elif uploaded_file.name.endswith(".docx"):
        return extract_text_from_docx(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Upload PDF or DOCX.")
