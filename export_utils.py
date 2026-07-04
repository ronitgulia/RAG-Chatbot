"""
Export utilities: export chat history as TXT, CSV, or PDF.
"""

import io
import csv
import json
import logging
from datetime import datetime
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def export_as_txt(messages: List[Dict[str, Any]], title: str = "Chat Export") -> bytes:
    """Export conversation as a formatted plain text file."""
    lines = [
        f"{title}",
        f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
    ]
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        timestamp = msg.get("timestamp", "")
        if timestamp:
            lines.append(f"[{timestamp}] {role}")
        else:
            lines.append(role)
        lines.append(content)
        lines.append("")
    return "\n".join(lines).encode("utf-8")


def export_as_csv(messages: List[Dict[str, Any]]) -> bytes:
    """Export conversation as CSV."""
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=["timestamp", "role", "content", "eval_score"],
        extrasaction="ignore",
    )
    writer.writeheader()
    for msg in messages:
        ev = msg.get("eval")
        eval_score = ev.overall_score if ev else ""
        writer.writerow({
            "timestamp": msg.get("timestamp", ""),
            "role": msg.get("role", ""),
            "content": msg.get("content", ""),
            "eval_score": eval_score,
        })
    return output.getvalue().encode("utf-8")


def export_as_json(messages: List[Dict[str, Any]]) -> bytes:
    """Export conversation as JSON."""
    from dataclasses import asdict, is_dataclass
    
    class DataclassEncoder(json.JSONEncoder):
        def default(self, obj):
            if is_dataclass(obj):
                return asdict(obj)
            return super().default(obj)
            
    return json.dumps(
        {"exported_at": datetime.now().isoformat(), "messages": messages},
        ensure_ascii=False,
        indent=2,
        cls=DataclassEncoder
    ).encode("utf-8")


def export_as_pdf(messages: List[Dict[str, Any]], title: str = "Chat Export") -> bytes:
    """Export conversation as PDF using fpdf2."""
    try:
        from fpdf import FPDF

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, title, ln=True, align="C")
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(0, 8, f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
        pdf.ln(4)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(4)

        for msg in messages:
            role = msg.get("role", "").upper()
            content = msg.get("content", "")

            if role == "USER":
                pdf.set_fill_color(240, 244, 255)
                pdf.set_font("Helvetica", "B", 10)
            else:
                pdf.set_fill_color(248, 248, 248)
                pdf.set_font("Helvetica", "B", 10)

            pdf.set_draw_color(220, 220, 220)
            pdf.cell(0, 7, role, ln=True, fill=True)
            pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(0, 5, content, border=0)
            pdf.ln(3)

        return bytes(pdf.output())

    except ImportError:
        logger.warning("fpdf2 not installed; falling back to TXT.")
        return export_as_txt(messages, title)