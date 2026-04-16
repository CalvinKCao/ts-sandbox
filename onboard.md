# Onboarding: Alliance Wiki Docs AI Assistant

## Purpose
This project provides a clean, AI-ready documentation set derived from the `docs.alliancecan.ca` MediaWiki dump. It's designed for AI assistants to easily navigate and understand Alliance Canada cluster policies, software, and workflows.

## Project Structure
- `docs.alliancecan.ca_mediawiki-20260319-wikidump/`: Raw/filtered XML files.
- `wiki_docs/`: Cleaned Markdown files (~240 pages focused on ML/AI, Python, and Slurm).
- `wikidump_to_markdown.py`: Conversion script (XML -> MD).
- `filter_wikidump.py`: XML-level filtering (stripping French, talk, etc.).

## Key Scripts
- `python3 wikidump_to_markdown.py`: Run this after generating a filtered XML to refresh the `wiki_docs/` folder.

## AI Assistant Usage
- Refer to `wiki_docs/` for specific software guides, Slurm policies, and cluster-specific details.
- Use internal links within the Markdown files to navigate related topics.
