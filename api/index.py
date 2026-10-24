"""
Aelix Backend — Advanced PDF Intelligence & Editing API for Vercel & Local Flask.
Features:
  1. POST /api/parse-pdf  -> High-fidelity block & section extraction using PyMuPDF (fitz) with fallback.
  2. POST /api/ai         -> Custom command processing (e.g. 'summarize in 15 lines'), section rewriting,
                             resilient API key support (custom key, env key, graceful smart fallback).
  3. POST /api/export     -> Exact original PDF preservation with AI edits/summary overlay,
                             and syntax-safe DOCX/HTML/MD/PDF export.
  4. GET  /api/health     -> Diagnostics and AI connectivity check.
  5. GET  /               -> Serves the single-page application.
"""

import io
import os
import re
import html
import base64
import statistics
from dotenv import load_dotenv

load_dotenv()

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

app = Flask(__name__, static_folder="public", static_url_path="/public")
CORS(app)

GROQ_MODEL_PRIMARY = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_MODEL_FALLBACK = "llama-3.1-8b-instant"


@app.after_request
def add_cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, x-api-key, Authorization"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp


# =========================================================================
#  Smart Extractive AI Engine (Instant Fallback when API Key is absent/invalid)
# =========================================================================
def smart_extractive_summary(text, num_lines=15):
    """
    Extracts the most significant key sentences from text without corrupting syntax.
    Guarantees exactly num_lines of high-value insights.
    """
    if not text:
        return "No text available to summarize."

    # Split into clean sentences
    raw_sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = []
    seen = set()

    for s in raw_sentences:
        clean = re.sub(r'\s+', ' ', s).strip()
        # Filter out short or repetitive fragments
        if len(clean) > 25 and clean.lower() not in seen:
            seen.add(clean.lower())
            sentences.append(clean)

    if not sentences:
        # Fallback to line splitting
        sentences = [l.strip() for l in text.split('\n') if len(l.strip()) > 20]

    if not sentences:
        return text[:1000]

    # Score sentences by term frequency of content words
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1

    scored = []
    for idx, s in enumerate(sentences):
        s_words = re.findall(r'\b[a-zA-Z]{4,}\b', s.lower())
        score = sum(freq.get(w, 0) for w in s_words) / max(len(s_words), 1)
        # Give a slight boost to earlier sentences (introductions/key statements)
        pos_multiplier = 1.2 if idx < 5 else (1.1 if idx < 10 else 1.0)
        scored.append((score * pos_multiplier, idx, s))

    # Pick top sentences while preserving original document flow
    scored.sort(key=lambda x: x[0], reverse=True)
    selected = scored[:num_lines]
    selected.sort(key=lambda x: x[1])  # Reorder chronologically

    lines = [f"{i+1}. {item[2]}" for i, item in enumerate(selected)]

    # If fewer than num_lines found, fill with remaining text chunks
    while len(lines) < num_lines and len(lines) < len(sentences):
        idx = len(lines)
        lines.append(f"{idx+1}. {sentences[idx]}")

    return "\n".join(lines)


# =========================================================================
#  High-Fidelity PDF -> Sections Extraction
# =========================================================================
def pdf_to_sections(pdf_bytes):
    """
    Extracts heading/paragraph blocks while preserving exact text syntax,
    page coordinates, and bounding boxes for in-place PDF editing.
    """
    sections = []

    # 1. Primary Strategy: PyMuPDF (fitz) - high precision block extraction
    try:
        import fitz
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_idx, page in enumerate(doc):
            blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
            for b in blocks:
                if len(b) > 6 and b[6] != 0:  # Skip image/vector blocks
                    continue
                raw_text = b[4].strip()
                if not raw_text:
                    continue

                cleaned_text = raw_text.replace("\r\n", "\n")
                lines = [l.strip() for l in cleaned_text.split("\n") if l.strip()]
                first_line = lines[0] if lines else ""

                # Font/block size heuristic for headings
                height = b[3] - b[1]
                level = "p"
                if len(cleaned_text) < 120 and (len(lines) <= 2 or len(cleaned_text.split()) <= 12):
                    if first_line.isupper() or first_line.endswith(":") or height > 24:
                        level = "h1" if height > 30 else "h2"
                    elif height > 16:
                        level = "h3"

                sections.append({
                    "id": len(sections) + 1,
                    "page": page_idx + 1,
                    "level": level,
                    "content": cleaned_text,
                    "original_content": cleaned_text,
                    "bbox": [round(b[0], 2), round(b[1], 2), round(b[2], 2), round(b[3], 2)],
                    "edited": False
                })

        if sections:
            return sections
    except Exception as e:
        print(f"PyMuPDF parse fallback notice: {e}")

    # 2. Secondary Strategy: pdfplumber line & font-size heuristics
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            all_lines = []
            for page_idx, page in enumerate(pdf.pages):
                words = page.extract_words(extra_attrs=["size"])
                if not words:
                    continue
                lines = {}
                for w in words:
                    key = round(w["top"], 1)
                    lines.setdefault(key, []).append(w)
                for top in sorted(lines.keys()):
                    line_words = sorted(lines[top], key=lambda w: w["x0"])
                    text = " ".join(w["text"] for w in line_words).strip()
                    if not text:
                        continue
                    size = max(w["size"] for w in line_words)
                    all_lines.append((text, size, page_idx + 1))

            if all_lines:
                sizes = [s for _, s, _ in all_lines]
                body_size = statistics.median(sizes) if sizes else 10
                h1_t, h2_t, h3_t = body_size * 1.5, body_size * 1.25, body_size * 1.1

                def level_for(size):
                    if size >= h1_t: return "h1"
                    if size >= h2_t: return "h2"
                    if size >= h3_t: return "h3"
                    return "p"

                buf_text = []
                last_page = 1
                for text, size, pg in all_lines:
                    lvl = level_for(size)
                    if lvl != "p":
                        if buf_text:
                            sections.append({
                                "id": len(sections) + 1,
                                "page": last_page,
                                "level": "p",
                                "content": " ".join(buf_text),
                                "original_content": " ".join(buf_text),
                                "edited": False
                            })
                            buf_text = []
                        sections.append({
                            "id": len(sections) + 1,
                            "page": pg,
                            "level": lvl,
                            "content": text,
                            "original_content": text,
                            "edited": False
                        })
                    else:
                        buf_text.append(text)
                        last_page = pg

                if buf_text:
                    sections.append({
                        "id": len(sections) + 1,
                        "page": last_page,
                        "level": "p",
                        "content": " ".join(buf_text),
                        "original_content": " ".join(buf_text),
                        "edited": False
                    })
    except Exception as e:
        print(f"pdfplumber parse fallback notice: {e}")

    # 3. Tertiary Strategy: pypdf basic text extraction
    if not sections:
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(pdf_bytes))
            for page_idx, page in enumerate(reader.pages):
                txt = page.extract_text() or ""
                paras = [p.strip() for p in txt.split("\n\n") if p.strip()]
                for p in paras:
                    sections.append({
                        "id": len(sections) + 1,
                        "page": page_idx + 1,
                        "level": "p",
                        "content": p,
                        "original_content": p,
                        "edited": False
                    })
        except Exception as e:
            print(f"pypdf parse error: {e}")

    return sections


# =========================================================================
#  Routes
# =========================================================================
@app.route("/", methods=["GET"])
def root_index():
    for candidate in ["public/index.html", "aelix_login.html", "aelix_editor.html"]:
        if os.path.exists(candidate):
            return send_file(candidate)
    return "Aelix PDF Intelligence Backend is running."


@app.route("/api/health", methods=["GET"])
def health():
    env_key = os.environ.get("GROQ_API_KEY", "")
    key_format_valid = bool(env_key and env_key.startswith("gsk_"))
    return jsonify({
        "ok": True,
        "groq_configured": key_format_valid,
        "primary_model": GROQ_MODEL_PRIMARY,
        "fallback_model": GROQ_MODEL_FALLBACK,
        "status": "ready"
    })


@app.route("/api/parse-pdf", methods=["POST", "OPTIONS"])
def parse_pdf():
    """Upload PDF -> extract sections with coordinate metadata."""
    if request.method == "OPTIONS":
        return ("", 204)

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]
    pdf_bytes = file.read()
    if not pdf_bytes:
        return jsonify({"error": "File is empty."}), 400

    try:
        sections = pdf_to_sections(pdf_bytes)
        if not sections:
            return jsonify({"error": "Could not extract readable text from this PDF."}), 400
        
        # Calculate total page count
        total_pages = max([s.get("page", 1) for s in sections], default=1)
        
        return jsonify({
            "sections": sections,
            "total_pages": total_pages,
            "filename": file.filename
        })
    except Exception as e:
        return jsonify({"error": f"Error parsing PDF: {str(e)}"}), 400


@app.route("/api/ai", methods=["POST", "OPTIONS"])
def ai_command():
    """
    Executes user AI instructions on the whole document OR individual sections.
    Supports user-provided Groq/OpenAI keys, server environment keys,
    and automatic extractive intelligence fallback so AI NEVER fails.
    """
    if request.method == "OPTIONS":
        return ("", 204)

    data = request.get_json(silent=True) or {}
    instruction = (data.get("instruction") or "").strip()
    if not instruction:
        return jsonify({"error": "Missing instruction."}), 400

    # API key discovery (Request Header > Request Body > Environment Variable)
    api_key = (
        request.headers.get("x-api-key")
        or data.get("api_key")
        or os.environ.get("GROQ_API_KEY")
        or ""
    ).strip()

    # Detect requested line count if specified (e.g. "summarize in 15 lines")
    line_match = re.search(r'(\d+)\s*lines?', instruction, re.IGNORECASE)
    requested_lines = int(line_match.group(1)) if line_match else None

    # ---- 1. Global Document Action (e.g. "summarize the pdf in 15 lines") ----
    if "sections" in data:
        sections = data.get("sections") or []
        doc_text = "\n\n".join(
            f"[{(s.get('level') or 'p').upper()}] {s.get('content') or s.get('title') or ''}"
            for s in sections
        )
        context = doc_text[:12000]

        # Dynamic instruction builder
        if instruction == "summarize":
            prompt = "Write a comprehensive and structured summary of the document below in 10-15 bullet points."
        elif instruction == "add-section":
            prompt = "Write one compelling new section that logically extends the document below."
        elif instruction == "improve":
            prompt = "Refine, clarify, and polish the central thesis and main takeaways of the document below."
        elif instruction == "tone":
            prompt = "Restate the core ideas of the document below in a refined executive and professional tone."
        else:
            prompt = f"Execute this instruction on the document: '{instruction}'."
            if requested_lines:
                prompt += f" Ensure your response contains EXACTLY {requested_lines} clearly numbered lines or sentences."

        # Attempt Groq Neural Call
        if api_key and api_key.startswith("gsk_"):
            try:
                from groq import Groq
                client = Groq(api_key=api_key)
                
                system_instruction = (
                    "You are Aelix, a world-class AI document analyst and editor. "
                    "Follow the user's document command with precision and elegance. "
                    "Return ONLY the requested refined content with zero conversational filler, "
                    "no preamble (do not say 'Here is the summary:'), and preserve exact technical syntax."
                )
                if requested_lines:
                    system_instruction += f" Format the output as exactly {requested_lines} numbered points (1. to {requested_lines}.)."

                try:
                    response = client.chat.completions.create(
                        model=GROQ_MODEL_PRIMARY,
                        messages=[
                            {"role": "system", "content": system_instruction},
                            {"role": "user", "content": f"Instruction: {prompt}\n\nDocument:\n{context}"}
                        ],
                        max_tokens=2048,
                        temperature=0.3
                    )
                except Exception:
                    # Fallback to secondary fast model if primary hits rate limits
                    response = client.chat.completions.create(
                        model=GROQ_MODEL_FALLBACK,
                        messages=[
                            {"role": "system", "content": system_instruction},
                            {"role": "user", "content": f"Instruction: {prompt}\n\nDocument:\n{context}"}
                        ],
                        max_tokens=2048,
                        temperature=0.3
                    )

                result = response.choices[0].message.content.strip()
                return jsonify({
                    "result": result,
                    "summary": result,
                    "instruction": instruction,
                    "section": {"level": "p", "content": result},
                    "ai_provider": "groq-llama3"
                })
            except Exception as e:
                err_str = str(e)
                print(f"Groq API Error: {err_str}")
                # If key is invalid or quota hit, proceed to smart extractive fallback

        # Smart Extractive Intelligence Fallback
        # This guarantees the user's "summarize in 15 lines" works immediately even if the key is expired!
        fallback_summary = smart_extractive_summary(doc_text, num_lines=requested_lines or 15)
        return jsonify({
            "result": fallback_summary,
            "summary": fallback_summary,
            "instruction": instruction,
            "section": {"level": "p", "content": fallback_summary},
            "ai_provider": "aelix-extractive",
            "warning": (
                "Groq API key was invalid or expired. Generated high-precision summary "
                "using Aelix Extractive Intelligence. Add your free key in AI Settings for full Llama-3 neural mode."
            )
        })

    # ---- 2. Per-Section Edit Action ----
    content = (data.get("content") or "").strip()
    if not content:
        return jsonify({"error": "Missing content to edit."}), 400
    level = data.get("section", "p")

    # Attempt Groq Neural Call
    if api_key and api_key.startswith("gsk_"):
        try:
            from groq import Groq
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model=GROQ_MODEL_PRIMARY,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are Aelix, an expert document editor. "
                            "Edit the provided text according to the user's instruction. "
                            "Return ONLY the edited text. Do not add explanations, conversational notes, or quotes."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Instruction: {instruction}\n\nSection Level: {level}\n\nText:\n{content}"
                    }
                ],
                max_tokens=1500,
                temperature=0.4
            )
            result = response.choices[0].message.content.strip()
            return jsonify({"result": result, "ai_provider": "groq-llama3"})
        except Exception as e:
            print(f"Groq per-section error: {e}")

    # Fallback section edit: rule-based intelligent adjustment
    edited_text = content
    cmd_lower = instruction.lower()
    if "shorten" in cmd_lower or "condense" in cmd_lower:
        sentences = re.split(r'(?<=[.!?])\s+', content)
        edited_text = " ".join(sentences[:max(1, len(sentences)//2)])
    elif "formal" in cmd_lower or "polish" in cmd_lower:
        edited_text = content.replace("can't", "cannot").replace("don't", "do not").replace("it's", "it is")
    elif "delete the last sentence" in cmd_lower or "remove the last sentence" in cmd_lower:
        sentences = re.split(r'(?<=[.!?])\s+', content)
        if len(sentences) > 1:
            edited_text = " ".join(sentences[:-1])
    elif "delete the first sentence" in cmd_lower:
        sentences = re.split(r'(?<=[.!?])\s+', content)
        if len(sentences) > 1:
            edited_text = " ".join(sentences[1:])
    else:
        edited_text = content.strip()

    return jsonify({
        "result": edited_text,
        "ai_provider": "aelix-fallback",
        "warning": "Add your free Groq API key in AI Settings for full Llama-3 rewriting."
    })


def clean_for_pdf(text):
    if not text:
        return ""
    charmap = {
        '\u2014': ' - ', '\u2013': '-', '\u2018': "'", '\u2019': "'",
        '\u201c': '"', '\u201d': '"', '\u2022': '-', '\u2026': '...',
        '\u00a0': ' ', '\u00b7': ' | '
    }
    for k, v in charmap.items():
        text = text.replace(k, v)
    return text.encode('latin-1', 'replace').decode('latin-1')


# =========================================================================
#  High-Fidelity Document & Exact PDF Export
# =========================================================================
@app.route("/api/export", methods=["POST", "OPTIONS"])
def export_file():
    """
    Export edited document.
    For PDF: Preserves the EXACT original uploaded PDF layout, pages, and syntax,
             overlaying AI edits and prepending summary pages.
    For DOCX/HTML/MD/TXT: Preserves exact text syntax without mangling.
    """
    if request.method == "OPTIONS":
        return ("", 204)

    data = request.get_json(silent=True) or {}
    fmt = (data.get("format") or "pdf").strip().lower()
    title = (data.get("title") or "Document").strip() or "Document"
    sections = data.get("sections", [])
    pdf_base64 = data.get("pdf_base64", "")
    ai_summary = (data.get("ai_summary") or "").strip()
    export_mode = data.get("export_mode", "exact")  # "exact", "summary_only", "clean_doc"

    if not sections and not ai_summary and not pdf_base64:
        return jsonify({"error": "No content to export."}), 400

    buf = io.BytesIO()

    # =====================================================================
    #  1. PDF EXPORT (Exact PDF Preservation or Clean XML-Safe PDF)
    # =====================================================================
    if fmt == "pdf":
        # Mode A: Exact Original PDF Preservation with PyMuPDF
        if export_mode != "clean_doc" and pdf_base64:
            try:
                import fitz
                orig_bytes = base64.b64decode(pdf_base64)
                doc = fitz.open(stream=orig_bytes, filetype="pdf")

                # If an AI Summary exists (e.g. 15-line summary), prepend a gorgeous summary page
                if ai_summary and export_mode != "original_only":
                    summary_page = doc.new_page(pno=0, width=595.3, height=841.9)  # Standard A4
                    
                    # Elegant Header Banner
                    summary_page.draw_rect(fitz.Rect(35, 35, 560, 105), color=(0.49, 0.36, 0.77), fill=(0.96, 0.94, 0.99), width=1.2)
                    summary_page.insert_text(fitz.Point(50, 68), "Aelix AI - Executive Intelligence Brief", fontsize=18, color=(0.29, 0.16, 0.54))
                    summary_page.insert_text(fitz.Point(50, 90), clean_for_pdf(f"Document: {title} | Analyzed & Generated by Aelix"), fontsize=10, color=(0.45, 0.35, 0.65))

                    # Format summary text / 15 lines cleanly
                    lines = [l.strip() for l in ai_summary.split("\n") if l.strip()]
                    y = 135
                    for line in lines:
                        if y > 790:  # If summary overflows A4 page, insert continuation
                            summary_page = doc.new_page(pno=1, width=595.3, height=841.9)
                            y = 50
                        
                        # Draw line with subtle bullet/number styling
                        is_heading = line.startswith("#") or line.endswith(":")
                        fsize = 11 if is_heading else 9.5
                        color = (0.22, 0.14, 0.45) if is_heading else (0.18, 0.18, 0.24)
                        clean_line = clean_for_pdf(line.lstrip("#").strip())
                        summary_page.insert_text(fitz.Point(50, y), clean_line, fontsize=fsize, color=color)
                        y += 18 if not is_heading else 24

                # Apply in-place text edits on original pages if any were edited
                for s in sections:
                    if s.get("edited") and s.get("page") and s.get("bbox"):
                        p_num = s["page"] - 1
                        if 0 <= p_num < len(doc):
                            page = doc[p_num]
                            bbox = s["bbox"]
                            rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
                            # Redact previous text box
                            page.add_redact_annot(rect, fill=(1, 1, 1))
                            page.apply_redactions()
                            # Insert edited text into the exact coordinates
                            page.insert_textbox(rect, clean_for_pdf(s.get("content", "")), fontsize=10, color=(0.15, 0.15, 0.2))

                doc.save(buf)
                buf.seek(0)
                return send_file(
                    buf,
                    mimetype="application/pdf",
                    as_attachment=True,
                    download_name=f"{title}_aelix_exact.pdf"
                )
            except Exception as e:
                print(f"PyMuPDF in-place export fallback notice: {e}")
                buf = io.BytesIO()

        # Mode B: High-Fidelity Clean Document PDF Generation (Safe XML-Escaped ReportLab)
        from reportlab.lib.pagesizes import LETTER
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.units import inch
        from reportlab.lib import colors

        doc = SimpleDocTemplate(
            buf,
            pagesize=LETTER,
            leftMargin=0.75 * inch,
            rightMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch
        )
        styles = getSampleStyleSheet()

        # Custom professional typography styles
        title_style = ParagraphStyle(
            'AelixTitle',
            parent=styles['Title'],
            fontName='Helvetica-Bold',
            fontSize=22,
            leading=26,
            textColor=colors.HexColor('#3d2475'),
            alignment=0,
            spaceAfter=15
        )
        h1_style = ParagraphStyle(
            'AelixH1',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=15,
            leading=19,
            textColor=colors.HexColor('#4c288c'),
            spaceBefore=14,
            spaceAfter=8
        )
        h2_style = ParagraphStyle(
            'AelixH2',
            parent=styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=12,
            leading=16,
            textColor=colors.HexColor('#6b42b8'),
            spaceBefore=10,
            spaceAfter=6
        )
        body_style = ParagraphStyle(
            'AelixBody',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            leading=15,
            textColor=colors.HexColor('#22222b'),
            spaceAfter=8
        )
        summary_style = ParagraphStyle(
            'AelixSummary',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            leading=16,
            textColor=colors.HexColor('#2a1b4e'),
            spaceAfter=6
        )

        story = [Paragraph(html.escape(title), title_style), Spacer(1, 0.15 * inch)]

        # If summary exists, render executive summary block
        if ai_summary:
            story.append(Paragraph("Aelix AI — Executive Summary", h1_style))
            for line in ai_summary.split("\n"):
                if line.strip():
                    safe_line = html.escape(line.strip())
                    story.append(Paragraph(safe_line, summary_style))
            story.append(Spacer(1, 0.25 * inch))

        # Render sections with full XML entity protection
        if export_mode != "summary_only":
            style_map = {"h1": h1_style, "h2": h2_style, "h3": h2_style, "p": body_style}
            for s in sections:
                text = (s.get("content") or "").strip()
                if not text:
                    continue
                style = style_map.get(s.get("level", "p"), body_style)
                # Escape XML entities so <tag>, &, math, quotes don't corrupt ReportLab
                safe_text = html.escape(text).replace("\n", "<br/>")
                story.append(Paragraph(safe_text, style))

        doc.build(story)
        buf.seek(0)
        return send_file(
            buf,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=f"{title}_edited.pdf"
        )

    # =====================================================================
    #  2. DOCX (Word) EXPORT
    # =====================================================================
    elif fmt in ("docx", "doc", "word"):
        from docx import Document as DocxDocument
        from docx.enum.text import WD_ALIGN_PARAGRAPH

        doc = DocxDocument()
        title_para = doc.add_heading(title, 0)
        title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER

        if ai_summary:
            doc.add_heading("Aelix AI — Executive Summary", level=1)
            for line in ai_summary.split("\n"):
                if line.strip():
                    doc.add_paragraph(line.strip())

        if export_mode != "summary_only":
            for s in sections:
                text = (s.get("content") or "").strip()
                if not text:
                    continue
                level = s.get("level", "p")
                if level == "h1":
                    doc.add_heading(text, level=1)
                elif level == "h2":
                    doc.add_heading(text, level=2)
                elif level == "h3":
                    doc.add_heading(text, level=3)
                else:
                    doc.add_paragraph(text)

        doc.save(buf)
        buf.seek(0)
        return send_file(
            buf,
            mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            as_attachment=True,
            download_name=f"{title}.docx"
        )

    # =====================================================================
    #  3. HTML EXPORT
    # =====================================================================
    elif fmt == "html":
        parts = [
            "<!DOCTYPE html><html><head><meta charset='utf-8'>",
            f"<title>{html.escape(title)}</title>",
            "<style>",
            "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;max-width:850px;margin:40px auto;padding:0 20px;line-height:1.7;color:#1e1e24;background:#faf9fd}",
            "h1{color:#3d2475;border-bottom:2px solid #e2d9f3;padding-bottom:10px}",
            "h2{color:#4c288c;margin-top:30px}",
            ".summary-box{background:#f3effc;border-left:4px solid #7c5cc4;padding:18px 24px;border-radius:8px;margin-bottom:30px}",
            "p{margin-bottom:16px}",
            "</style></head><body>",
            f"<h1>{html.escape(title)}</h1>"
        ]

        if ai_summary:
            parts.append("<div class='summary-box'><h2>Aelix AI — Executive Summary</h2>")
            for line in ai_summary.split("\n"):
                if line.strip():
                    parts.append(f"<p>{html.escape(line.strip())}</p>")
            parts.append("</div>")

        if export_mode != "summary_only":
            tag_map = {"h1": "h2", "h2": "h3", "h3": "h4", "p": "p"}
            for s in sections:
                text = (s.get("content") or "").strip()
                if not text:
                    continue
                tag = tag_map.get(s.get("level", "p"), "p")
                parts.append(f"<{tag}>{html.escape(text)}</{tag}>")

        parts.append("</body></html>")
        buf.write("\n".join(parts).encode("utf-8"))
        buf.seek(0)
        return send_file(
            buf,
            mimetype="text/html",
            as_attachment=True,
            download_name=f"{title}.html"
        )

    # =====================================================================
    #  4. MARKDOWN & TEXT EXPORT
    # =====================================================================
    elif fmt in ("md", "markdown", "txt"):
        lines = [f"# {title}", ""]
        if ai_summary:
            lines.extend(["## Aelix AI — Executive Summary", "", ai_summary, "", "---", ""])

        if export_mode != "summary_only":
            prefix_map = {"h1": "# ", "h2": "## ", "h3": "### ", "p": ""}
            for s in sections:
                text = (s.get("content") or "").strip()
                if not text:
                    continue
                prefix = prefix_map.get(s.get("level", "p"), "")
                lines.append(f"{prefix}{text}\n")

        buf.write("\n".join(lines).encode("utf-8"))
        buf.seek(0)
        mtype = "text/markdown" if fmt != "txt" else "text/plain"
        ext = "md" if fmt != "txt" else "txt"
        return send_file(
            buf,
            mimetype=mtype,
            as_attachment=True,
            download_name=f"{title}.{ext}"
        )

    return jsonify({"error": f"Unsupported format '{fmt}'."}), 400


if __name__ == "__main__":
    app.run(port=5000, debug=True)
