# Alliance Wiki Docs AI Assistant

This project processes a MediaWiki XML dump from `docs.alliancecan.ca` into a clean, hierarchical Markdown format optimized for AI coding assistants.

## Architecture

- **`docs.alliancecan.ca_mediawiki-20260319-wikidump/`**: Contains the raw and filtered XML dumps.
- **`wikidump_to_markdown.py`**: The core conversion script. It parses the XML, handles language preferences (preferring `/en`), cleans wikitext (stripping `<translate>`, `<!--T:123-->`, etc.), and converts MediaWiki templates (`{{File}}`, `{{Command}}`) into standard Markdown.
- **`wiki_docs/`**: The output directory containing the processed Markdown files. It maintains a shallow hierarchy based on wiki subpages.

## Data Flow
1. **Raw XML** -> `filter_wikidump.py` -> **Filtered XML**
2. **Filtered XML** -> `wikidump_to_markdown.py` -> **Clean Markdown Files**
