#!/usr/bin/env python3
import os
import re
import sys
import html
from pathlib import Path
import xml.etree.ElementTree as ET

MW_NS = "http://www.mediawiki.org/xml/export-0.11/"
MW = f"{{{MW_NS}}}"

def clean_wikitext(text):
    if not text:
        return ""
    
    # Remove redirects
    if text.lstrip().upper().startswith("#REDIRECT"):
        m = re.search(r'\[\[([^\]]+)\]\]', text)
        if m:
            target = m.group(1).strip().replace(' ', '_')
            if target.endswith('/en'): target = target[:-3]
            return f"See [{m.group(1).strip()}]({target}.md)"
        return ""
    
    # Remove translation tags
    text = re.sub(r'</?translate>', '', text)
    text = re.sub(r'<languages\s*/>', '', text)
    text = re.sub(r'<!--T:\d+-->', '', text)
    
    # Strip <nowiki> tags
    text = re.sub(r'<nowiki>(.*?)</nowiki>', r'\1', text, flags=re.DOTALL)
    
    # Bold/Italic
    text = re.sub(r"'''(.*?)'''", r"**\1**", text)
    text = re.sub(r"''(.*?)''", r"*\1*", text)
    
    # Convert headers: == H2 == -> ## H2, === H3 === -> ### H3
    text = re.sub(r'^======\s*(.*?)\s*======', r'###### \1', text, flags=re.MULTILINE)
    text = re.sub(r'^=====\s*(.*?)\s*=====', r'##### \1', text, flags=re.MULTILINE)
    text = re.sub(r'^====\s*(.*?)\s*====', r'#### \1', text, flags=re.MULTILINE)
    text = re.sub(r'^===\s*(.*?)\s*===', r'### \1', text, flags=re.MULTILINE)
    text = re.sub(r'^==\s*(.*?)\s*==', r'## \1', text, flags=re.MULTILINE)
    
    # Simple template conversions
    def handle_file_template(m):
        content = m.group(1)
        name_match = re.search(r'name\s*=\s*([^|\n]+)', content)
        lang_match = re.search(r'lang\s*=\s*([^|\n]+)', content)
        body_match = re.search(r'contents\s*=\s*(.*)', content, re.DOTALL)
        
        name = name_match.group(1).strip() if name_match else ""
        lang = lang_match.group(1).strip().strip('"') if lang_match else ""
        body = body_match.group(1).strip() if body_match else ""
        
        header = f"**File: {name}**\n" if name else ""
        return f"{header}```{lang}\n{body}\n```"

    text = re.sub(r'\{\{File\s*\|(.*?)\}\}', handle_file_template, text, flags=re.DOTALL)
    
    def handle_commands_template(m):
        body = m.group(1)
        body = re.sub(r'\|\s*prompt=[^|]*', '', body)
        body = re.sub(r'^\s*\|\s*', '', body, flags=re.MULTILINE)
        return f"```bash\n{body.strip()}\n```"
        
    text = re.sub(r'\{\{Commands?\s*\|(.*?)\}\}', handle_commands_template, text, flags=re.DOTALL)
    
    # Convert <source lang="bash">...</source> to ```bash\n...\n```
    def handle_source_tag(m):
        lang = m.group(1) or ""
        body = m.group(2).strip()
        return f"```{lang}\n{body}\n```"
    text = re.sub(r'<source\s+lang="?([^">]*)"?>(.*?)</source>', handle_source_tag, text, flags=re.DOTALL)
    
    # Convert <pre>...</pre> to ```\n...\n```
    text = re.sub(r'<pre>(.*?)</pre>', r'```\n\1\n```', text, flags=re.DOTALL)

    # Strip simple templates like {{draft}}, {{note}}, etc.
    text = re.sub(r'\{\{(?:draft|note|warning|info|tip|important|caution|see also)\s*.*?\}\}', '', text, flags=re.IGNORECASE)
    
    # Internal links normalization: [[Target|Text]] -> [Text](Target.md)
    def normalize_target(target):
        target = target.strip()
        if target.endswith('/en'): target = target[:-3]
        return target.replace(' ', '_')

    def handle_internal_link(m):
        target = normalize_target(m.group(1))
        label = m.group(2).strip()
        return f"[{label}]({target}.md)"
    
    text = re.sub(r'\[\[([^\]|]+)\|([^\]]+)\]\]', handle_internal_link, text)
    
    def handle_simple_internal_link(m):
        raw_target = m.group(1).strip()
        target = normalize_target(raw_target)
        return f"[{raw_target}]({target}.md)"
        
    text = re.sub(r'\[\[([^\]|]+)\]\]', handle_simple_internal_link, text)
    
    # Strip Category links lines (often at top/bottom)
    text = re.sub(r'^\[Category:[^\]]+\]\(Category:[^\]]+\.md\)\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\[Category:[^\]]+\]\(Category:[^\]]+\.md\)', '', text, flags=re.MULTILINE)
    
    # External links: [http://foo Text] -> [Text](http://foo)
    text = re.sub(r'\[(https?://[^\s\]]+)\s+([^\]]+)\]', r'[\2](\1)', text)
    text = re.sub(r'\[(https?://[^\s\]]+)\]', r'<\1>', text)
    
    text = re.sub(r'^\*\s+', r'- ', text, flags=re.MULTILINE)
    text = re.sub(r'^#\s+', r'1. ', text, flags=re.MULTILINE)
    
    return text.strip()

def main():
    src_xml = Path("docs.alliancecan.ca_mediawiki-20260319-wikidump/docs.alliancecan.ca_mediawiki-20260325-current-filtered.xml")
    out_dir = Path("wiki_docs")
    if out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)
    out_dir.mkdir(exist_ok=True)
    
    print(f"Parsing {src_xml}...")
    context = ET.iterparse(str(src_xml), events=("end",))
    
    pages = {}
    
    for event, elem in context:
        if elem.tag == f"{MW}page":
            title_el = elem.find(f"{MW}title")
            if title_el is None: continue
            title = (title_el.text or "").strip()
            
            if any(title.startswith(prefix + ":") for prefix in ["Category", "Template", "File", "MediaWiki", "Talk"]):
                elem.clear()
                continue
            
            rev = elem.find(f"{MW}revision")
            if rev is None: 
                elem.clear()
                continue
            text_el = rev.find(f"{MW}text")
            if text_el is None: 
                elem.clear()
                continue
            
            content = text_el.text or ""
            
            clean_title = title
            is_en = False
            if title.endswith("/en"):
                clean_title = title[:-3]
                is_en = True
            
            # If it's a redirect, we'll store it but mark it
            is_redirect = content.lstrip().upper().startswith("#REDIRECT")
            
            if is_en or clean_title not in pages:
                # Priority: /en non-redirect > non-/en non-redirect > /en redirect > non-/en redirect
                if clean_title in pages:
                    existing_is_redirect = pages[clean_title][2]
                    existing_is_en = pages[clean_title][3]
                    
                    # If existing is already a non-redirect /en, keep it
                    if not existing_is_redirect and existing_is_en:
                        elem.clear()
                        continue
                        
                    # If this is a redirect and existing is not, skip this
                    if is_redirect and not existing_is_redirect:
                        elem.clear()
                        continue
                
                pages[clean_title] = (title, content, is_redirect, is_en)
            
            elem.clear()

    print(f"Total unique pages found: {len(pages)}")
    
    for clean_title, (orig_title, content, is_redirect, is_en) in pages.items():
        if is_redirect and len(pages) > 100: # Heuristic to skip redirects if we have plenty of real content
            # Only keep redirects if they don't have a clean path already
            continue

        safe_path_parts = [p.replace(' ', '_') for p in clean_title.split('/')]
        file_path = out_dir.joinpath(*safe_path_parts).with_suffix(".md")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        md_content = clean_wikitext(content)
        if not md_content and is_redirect: continue
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"# {clean_title}\n\n")
            f.write(md_content)
            
    print(f"Conversion complete! Markdown files are in {out_dir}/")

if __name__ == "__main__":
    main()
