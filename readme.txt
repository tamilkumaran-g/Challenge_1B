1B/
├── Collection_1/
│   ├── challenge1b_input.json      ✅ Required
│   ├── pdfs/                       ✅ Folder of PDFs
│   │   ├── File1.pdf
│   │   ├── File2.pdf
│   └── challenge1b_output.json     ⬅ Generated automatically
├── Collection_2/
│   ├── challenge1b_input.json
│   ├── pdfs/
│   └── challenge1b_output.json
├── parsed_json/                   ⬅ Generated automatically for intermediate headings
│   ├── Collection_1/
│   │   ├── File1.json             ⬅ structured titles from File1.pdf
│   │   └── ...
│   └── Collection_2/

Usage: python main_pipeline.py /path/to/input-root-folder