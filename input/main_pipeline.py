import os
import sys
import json
import datetime
import fitz
import re
import numpy as np
from sentence_transformers import SentenceTransformer, util

# IMPORTANT:
# This script expects process_pdfs.py to be importable and provide a `process_pdf(pdf_path)` function.
# You must have process_pdfs.py in the PYTHONPATH or same folder.

from input.process_pdfs import process_pdf

def process_all_pdfs_in_folder(pdf_folder, parsed_json_folder):
    """
    Process all PDFs inside pdf_folder (non-recursive) and save parsed JSONs
    to parsed_json_folder.
    """
    if not os.path.exists(parsed_json_folder):
        os.makedirs(parsed_json_folder)

    for file in os.listdir(pdf_folder):
        if file.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, file)
            try:
                parsed_data = process_pdf(pdf_path)
            except Exception as e:
                print(f"Error processing {pdf_path}: {e}")
                continue
            out_file = os.path.join(parsed_json_folder, f"{os.path.splitext(file)[0]}.json")
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(parsed_data, f, indent=2, ensure_ascii=False)
            print(f"Parsed JSON saved: {out_file}")

def extract_text_for_page(pdf_path, page_number):
    doc = fitz.open(pdf_path)
    if 1 <= page_number <= len(doc):
        return doc[page_number - 1].get_text("text").strip()
    return ""

def build_sections_from_parsed_json(pdf_filename, parsed_json_path, pdf_path):
    try:
        with open(parsed_json_path, "r", encoding="utf-8") as f:
            parsed_data = json.load(f)
    except Exception as e:
        print(f"Could not load parsed JSON {parsed_json_path}: {e}")
        parsed_data = {}

    outline = parsed_data.get("outline", [])

    sections = []
    if not outline:
        # fallback: page-based sections
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            text = doc[page_num].get_text("text").strip()
            if text:
                sections.append({
                    "document": pdf_filename,
                    "page_number": page_num + 1,
                    "text": text,
                    "section_title": f"Page {page_num + 1}"
                })
        return sections

    for heading in outline:
        page_num = heading.get("page")
        title = heading.get("text")
        if page_num is None or title is None:
            continue
        text = extract_text_for_page(pdf_path, page_num)
        if not text:
            continue
        sections.append({
            "document": pdf_filename,
            "page_number": page_num,
            "text": text,
            "section_title": title
        })

    return sections

def create_summary(text, num_sentences=3):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    selected = [s.strip() for s in sentences if s.strip()][:num_sentences]
    return " ".join(selected)

def process_documents(input_data, pdf_folder, parsed_json_folder, top_n=5):
    persona = input_data.get("persona", {}).get("role", "")
    job = input_data.get("job_to_be_done", {}).get("task", "")
    documents = input_data.get("documents", [])

    instruction = f"{persona} {job}"

    print(f"Persona: {persona}")
    print(f"Job to be done: {job}")
    print(f"Processing {len(documents)} documents...")

    all_sections = []
    for doc in documents:
        pdf_filename = doc.get("filename")
        if not pdf_filename:
            continue

        pdf_path = os.path.join(pdf_folder, "pdfs", pdf_filename)
        parsed_json_path = os.path.join(parsed_json_folder, pdf_filename)
        parsed_json_path = parsed_json_path.replace('.pdf', '.json')

        if not os.path.isfile(pdf_path):
            print(f"Warning: PDF file not found: {pdf_path}")
            continue
        if not os.path.isfile(parsed_json_path):
            print(f"Warning: Parsed JSON not found for {pdf_filename}. Falling back to page-based sections.")
            doc_pdf = fitz.open(pdf_path)
            for i in range(len(doc_pdf)):
                text = doc_pdf[i].get_text("text").strip()
                if text:
                    all_sections.append({
                        "document": pdf_filename,
                        "page_number": i + 1,
                        "text": text,
                        "section_title": f"Page {i + 1}"
                    })
            continue

        sections = build_sections_from_parsed_json(pdf_filename, parsed_json_path, pdf_path)
        if not sections:
            print(f"No sections found in parsed JSON for {pdf_filename}, falling back.")
            doc_pdf = fitz.open(pdf_path)
            for i in range(len(doc_pdf)):
                text = doc_pdf[i].get_text("text").strip()
                if text:
                    all_sections.append({
                        "document": pdf_filename,
                        "page_number": i + 1,
                        "text": text,
                        "section_title": f"Page {i + 1}"
                    })
        else:
            all_sections.extend(sections)

    if not all_sections:
        raise RuntimeError("No sections extracted from PDFs.")

    print(f"Total sections extracted: {len(all_sections)}")

    model = SentenceTransformer('all-mpnet-base-v2')

    print("Computing embeddings and ranking sections...")
    query_emb = model.encode([instruction], convert_to_tensor=True)
    corpus_embs = model.encode([s["text"] for s in all_sections], convert_to_tensor=True)
    cos_scores = util.cos_sim(query_emb, corpus_embs).squeeze()
    ranked_indices = np.argsort(-cos_scores.cpu().numpy())

    top_indices = ranked_indices[:top_n]

    output = {
        "metadata": {
            "input_documents": [doc.get("filename") for doc in documents],
            "persona": persona,
            "job_to_be_done": job,
            "processing_timestamp": datetime.datetime.now().isoformat()
        },
        "extracted_sections": [],
        "subsection_analysis": []
    }

    for rank, idx in enumerate(top_indices, 1):
        section = all_sections[idx]
        summary = create_summary(section["text"])
        output["extracted_sections"].append({
            "document": section["document"],
            "section_title": section["section_title"],
            "importance_rank": rank,
            "page_number": section["page_number"]
        })
        output["subsection_analysis"].append({
            "document": section["document"],
            "refined_text": summary,
            "page_number": section["page_number"]
        })

    # Fill with None if fewer than top_n
    while len(output["extracted_sections"]) < top_n:
        i = len(output["extracted_sections"]) + 1
        output["extracted_sections"].append({
            "document": None,
            "section_title": None,
            "importance_rank": i,
            "page_number": None
        })
        output["subsection_analysis"].append({
            "document": None,
            "refined_text": None,
            "page_number": None
        })

    return output

def main():
    if len(sys.argv) != 2:
        print("Usage: python main_pipeline.py /path/to/input-root-folder")
        sys.exit(1)

    root_folder = sys.argv[1]

    if not os.path.isdir(root_folder):
        print("Invalid root folder path.")
        sys.exit(1)

    parsed_json_root = os.path.join(root_folder, "parsed_json")
    if not os.path.exists(parsed_json_root):
        os.makedirs(parsed_json_root)

    # Find all subfolders inside root folder (*only directories with challenge1b_input.json)
    for subfolder_name in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder_name)
        if not os.path.isdir(subfolder_path):
            continue

        input_json_path = os.path.join(subfolder_path, "challenge1b_input.json")
        pdfs_folder = os.path.join(subfolder_path, "pdfs")
        output_json_path = os.path.join(subfolder_path, "challenge1b_output.json")
        parsed_json_subfolder = os.path.join(parsed_json_root, subfolder_name)

        if not os.path.isfile(input_json_path):
            print(f"Skipping {subfolder_name}: No challenge1b_input.json found.")
            continue
        if not os.path.isdir(pdfs_folder):
            print(f"Skipping {subfolder_name}: No pdfs/ folder found.")
            continue

        print(f"\n=== Processing folder: {subfolder_name} ===")

        # Step1: Process PDFs and save parsed json in parsed_json/<subfolder>/
        process_all_pdfs_in_folder(pdfs_folder, parsed_json_subfolder)

        # Step2: Load input JSON
        with open(input_json_path, "r", encoding="utf-8") as f:
            input_data = json.load(f)

        # Step3: Run main processing pipeline
        output_data = process_documents(input_data, subfolder_path, parsed_json_subfolder, top_n=5)

        # Step 4: Save output JSON
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"Output JSON saved at {output_json_path}")

if __name__ == "__main__":
    main()
