# Architecture Log

## 2026-04-16
- Initial project setup.
- Developed `wikidump_to_markdown.py` to convert MediaWiki XML to clean Markdown.
- Created `wiki_docs/` with normalized English-first content.
- Stripped MediaWiki-specific tags (`<translate>`, `<!--T:...-->`) and converted templates (`{{File}}`, `{{Commands}}`, `<source>`, `<pre>`) to Markdown code blocks.
- Filtered down from 600+ to ~240 files to keep only ML/AI, Python, Slurm, and cluster-specific technical documentation.
