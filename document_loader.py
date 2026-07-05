"""
Document loading utilities for PDF, DOCX, and TXT files.
"""

import io
import os
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


def load_pdf(file_obj) -> List[Dict[str, Any]]:
    """Load and extract text from a PDF file using pypdf with fallback to PyPDF2."""
    documents = []
    file_name = getattr(file_obj, "name", "unknown.pdf")
    file_bytes = file_obj.read() if hasattr(file_obj, "read") else file_obj

    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                documents.append({
                    "page_content": text.strip(),
                    "metadata": {
                        "source": file_name,
                        "page": page_num + 1,
                        "file_type": "pdf",
                        "total_pages": len(reader.pages),
                    },
                })
    except Exception as e1:
        logger.warning(f"pypdf failed ({e1}), trying PyPDF2...")
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    documents.append({
                        "page_content": text.strip(),
                        "metadata": {
                            "source": file_name,
                            "page": page_num + 1,
                            "file_type": "pdf",
                            "total_pages": len(reader.pages),
                        },
                    })
        except Exception as e2:
            raise RuntimeError(f"Failed to load PDF: {e1} | {e2}") from e2

    return documents


def load_docx(file_obj) -> List[Dict[str, Any]]:
    """Load and extract text from a DOCX file, including tables and paragraphs."""
    from docx import Document
    file_name = getattr(file_obj, "name", "unknown.docx")
    file_bytes = file_obj.read() if hasattr(file_obj, "read") else file_obj

    doc = Document(io.BytesIO(file_bytes))
    paragraphs = [para.text.strip() for para in doc.paragraphs if para.text.strip()]

    # Extract tables into structured readable rows
    table_lines = []
    for table in getattr(doc, "tables", []):
        for row in table.rows:
            row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if row_cells:
                table_lines.append(" | ".join(row_cells))

    content_parts = []
    if paragraphs:
        content_parts.append("\n\n".join(paragraphs))
    if table_lines:
        content_parts.append("[Tables]\n" + "\n".join(table_lines))

    full_text = "\n\n".join(content_parts).strip()
    if not full_text:
        return []

    return [{
        "page_content": full_text,
        "metadata": {
            "source": file_name,
            "file_type": "docx",
            "paragraph_count": len(paragraphs),
            "table_count": len(getattr(doc, "tables", [])),
        },
    }]


def load_csv(file_obj) -> List[Dict[str, Any]]:
    """Load tabular data from a CSV file."""
    import csv
    file_name = getattr(file_obj, "name", "unknown.csv")
    raw = file_obj.read() if hasattr(file_obj, "read") else file_obj
    if isinstance(raw, bytes):
        text = raw.decode("utf-8", errors="replace")
    else:
        text = str(raw)

    reader = csv.DictReader(io.StringIO(text))
    rows = []
    for idx, row in enumerate(reader):
        row_str = ", ".join(f"{k}: {v}" for k, v in row.items() if v)
        if row_str.strip():
            rows.append(f"Row {idx + 1}: {row_str}")

    if not rows:
        return []

    return [{
        "page_content": "\n".join(rows),
        "metadata": {
            "source": file_name,
            "file_type": "csv",
            "row_count": len(rows),
        },
    }]


def load_json(file_obj) -> List[Dict[str, Any]]:
    """Load structured data from a JSON file."""
    import json
    file_name = getattr(file_obj, "name", "unknown.json")
    raw = file_obj.read() if hasattr(file_obj, "read") else file_obj
    if isinstance(raw, bytes):
        text = raw.decode("utf-8", errors="replace")
    else:
        text = str(raw)

    try:
        data = json.loads(text)
    except Exception as e:
        logger.warning(f"JSON parsing error for {file_name}: {e}")
        return load_txt(file_obj)

    if isinstance(data, list):
        items = [json.dumps(item, ensure_ascii=False) for item in data if item]
        content = "\n".join(items)
    elif isinstance(data, dict):
        content = json.dumps(data, indent=2, ensure_ascii=False)
    else:
        content = str(data)

    if not content.strip():
        return []

    return [{
        "page_content": content.strip(),
        "metadata": {
            "source": file_name,
            "file_type": "json",
        },
    }]


def load_txt(file_obj) -> List[Dict[str, Any]]:
    """Load and extract text from a TXT file."""
    file_name = getattr(file_obj, "name", "unknown.txt")
    raw = file_obj.read() if hasattr(file_obj, "read") else file_obj
    if isinstance(raw, bytes):
        text = raw.decode("utf-8", errors="replace")
    else:
        text = raw

    if not text.strip():
        return []

    return [{
        "page_content": text.strip(),
        "metadata": {
            "source": file_name,
            "file_type": "txt",
            "char_count": len(text),
        },
    }]


def load_document(file_obj) -> List[Dict[str, Any]]:
    """Dispatch loader based on file extension."""
    file_name = getattr(file_obj, "name", "")
    ext = Path(file_name).suffix.lower()

    loaders = {
        ".pdf": load_pdf,
        ".docx": load_docx,
        ".doc": load_docx,
        ".txt": load_txt,
        ".md": load_txt,
        ".csv": load_csv,
        ".json": load_json,
    }

    loader = loaders.get(ext)
    if loader is None:
        raise ValueError(f"Unsupported file type: '{ext}'. Supported: {list(loaders.keys())}")

    docs = loader(file_obj)
    logger.info(f"Loaded {len(docs)} document section(s) from '{file_name}'.")
    return docs


def load_multiple_documents(files: list) -> List[Dict[str, Any]]:
    """Load multiple files and aggregate the results."""
    all_docs = []
    errors = []

    for f in files:
        try:
            docs = load_document(f)
            all_docs.extend(docs)
        except Exception as e:
            errors.append(f"{getattr(f, 'name', 'file')}: {e}")
            logger.error(f"Error loading file: {e}")

    if errors:
        logger.warning(f"Errors during loading: {errors}")

    return all_docs, errors