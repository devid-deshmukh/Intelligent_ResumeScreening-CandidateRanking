import io

import pandas as pd
from docx import Document

from src.ranker import rank_resumes
from src.uploads import extract_resume_text, process_uploaded_resumes


class UploadedFile:
    def __init__(self, name: str, content: bytes):
        self.name = name
        self._content = content

    def getvalue(self) -> bytes:
        return self._content


def docx_bytes(text: str) -> bytes:
    document = Document()
    document.add_paragraph(text)
    output = io.BytesIO()
    document.save(output)
    return output.getvalue()


def pdf_bytes(text: str) -> bytes:
    content = f"BT /F1 12 Tf 20 100 Td ({text}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 300] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(content)).encode("ascii") + b" >>\nstream\n" + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{index} 0 obj\n".encode("ascii"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")
    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(f"trailer\n<< /Root 1 0 R /Size {len(objects) + 1} >>\nstartxref\n{xref_offset}\n%%EOF".encode("ascii"))
    return bytes(pdf)


def test_extract_docx_and_normalize_resume():
    text = extract_resume_text("candidate.docx", docx_bytes("Senior Python Engineer\n8 years of experience"))
    resumes, errors = process_uploaded_resumes(
        [UploadedFile("candidate.docx", docx_bytes("Senior Python Engineer\n8 years of experience\nPython SQL"))],
        {"must_have_skills": ["Python", "SQL"], "nice_to_have_skills": []},
    )
    assert text.startswith("Senior Python Engineer")
    assert errors == []
    assert resumes.iloc[0]["years_experience"] == 8
    assert resumes.iloc[0]["skills"] == ["python", "sql"]


def test_upload_processing_reports_empty_and_unsupported_files():
    resumes, errors = process_uploaded_resumes(
        [UploadedFile("empty.pdf", b""), UploadedFile("notes.txt", b"hello")],
        {"must_have_skills": [], "nice_to_have_skills": []},
    )
    assert isinstance(resumes, pd.DataFrame)
    assert resumes.empty
    assert len(errors) == 2
    assert all(error["error"] for error in errors)


def test_duplicate_filenames_get_unique_resume_ids():
    files = [
        UploadedFile("candidate.docx", docx_bytes("Python")),
        UploadedFile("candidate.docx", docx_bytes("SQL")),
    ]
    resumes, errors = process_uploaded_resumes(files, {"must_have_skills": ["Python"], "nice_to_have_skills": []})
    assert errors == []
    assert resumes["resume_id"].tolist() == ["candidate", "candidate_2"]


def test_multiple_file_uploads_rank_every_valid_candidate():
    job = {
        "job_title": "Python Engineer",
        "description": "Build services with Python and SQL",
        "must_have_skills": ["Python", "SQL"],
        "nice_to_have_skills": ["Docker"],
    }
    files = [
        UploadedFile("strong.docx", docx_bytes("Senior Python Engineer\n8 years of experience\nPython SQL Docker")),
        UploadedFile("other.pdf", pdf_bytes("Junior Analyst\n1 year of experience\nPython")),
        UploadedFile("broken.docx", b"not a docx"),
    ]
    resumes, errors = process_uploaded_resumes(files, job)
    ranked = rank_resumes(job, top_k=10, method="combined", resumes=resumes)

    assert len(resumes) == len(ranked) == 2
    assert len(errors) == 1
    assert ranked["score"].notna().all()
    assert ranked["score"].is_monotonic_decreasing
    assert ranked.iloc[0]["filename"] == "strong.docx"
    assert ranked.iloc[0]["matched_skills"] == ["python", "sql", "docker"]
    assert ranked.iloc[1]["matched_skills"] == ["python"]
