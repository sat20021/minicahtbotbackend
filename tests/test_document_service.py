"""
Tests for document extraction (PDF and TXT) and validation.
"""
import pytest
import fitz
from app.document_service import (
    extract_text_from_txt,
    extract_text_from_pdf,
    extract_text,
    DocumentProcessingError,
)


def create_minimal_pdf_bytes(text: str) -> bytes:
    """Helper to generate a valid in-memory PDF byte stream using PyMuPDF."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), text)
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes


def test_txt_extraction_valid():
    content = "Hello, world! This is a simple test document.\nWith two lines.".encode("utf-8")
    extracted = extract_text_from_txt(content)
    assert "Hello, world!" in extracted
    assert "With two lines." in extracted


def test_txt_extraction_empty():
    with pytest.raises(DocumentProcessingError, match="empty"):
        extract_text_from_txt(b"")

    with pytest.raises(DocumentProcessingError, match="whitespace"):
        extract_text_from_txt(b"   \n\t  ")


def test_pdf_extraction_valid():
    sample_text = "This is PyMuPDF extracting text from an in-memory PDF document."
    pdf_bytes = create_minimal_pdf_bytes(sample_text)
    extracted = extract_text_from_pdf(pdf_bytes)
    assert sample_text in extracted


def test_pdf_extraction_empty():
    with pytest.raises(DocumentProcessingError, match="empty"):
        extract_text_from_pdf(b"")


def test_pdf_extraction_corrupt():
    with pytest.raises(DocumentProcessingError, match="Corrupt or invalid PDF"):
        extract_text_from_pdf(b"%PDF-1.4 definitely not a valid pdf stream 12345")


def test_extract_text_router():
    txt_bytes = "Plain text content".encode("utf-8")
    text, file_type = extract_text("doc.txt", txt_bytes)
    assert file_type == "txt"
    assert text == "Plain text content"

    pdf_bytes = create_minimal_pdf_bytes("PDF content")
    text, file_type = extract_text("manual.pdf", pdf_bytes)
    assert file_type == "pdf"
    assert "PDF content" in text

    with pytest.raises(DocumentProcessingError, match="Unsupported file format"):
        extract_text("image.png", b"\x89PNG...")
