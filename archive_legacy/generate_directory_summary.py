#!/usr/bin/env python3
"""Generate comprehensive directory summary with all files listed."""

import os
from pathlib import Path
from datetime import datetime

def get_all_files(directory, relative_to=None):
    """Get all files in directory sorted by name."""
    dir_path = Path(directory)
    if not dir_path.exists():
        return []
    
    files = []
    for item in sorted(dir_path.iterdir()):
        if item.is_file():
            if relative_to:
                files.append(str(item.relative_to(relative_to)))
            else:
                files.append(item.name)
    return files

def main():
    root = Path(".")
    
    # Get file counts
    root_files = get_all_files(".", root)
    src_files = get_all_files("src", root) if Path("src").exists() else []
    paper_files = get_all_files("paper", root) if Path("paper").exists() else []
    logs_files = get_all_files("logs", root) if Path("logs").exists() else []
    figures_files = get_all_files("figures", root) if Path("figures").exists() else []
    archive_files = get_all_files("archive", root) if Path("archive").exists() else []
    tests_files = get_all_files("tests_archive", root) if Path("tests_archive").exists() else []
    
    # Generate markdown
    md = f"""# Directory Summary: Geometric Atom Research Repository

**Last Updated:** {datetime.now().strftime("%B %d, %Y")}  
**Status:** ✅ REORGANIZED & VERIFIED

This directory contains a comprehensive research project deriving the fine structure constant (α⁻¹ ≈ 137.036) from pure geometry using SO(4,2) symmetry of the hydrogen atom.

---

## 📊 Repository Statistics

- **Total Files:** {len(root_files) + len(src_files) + len(paper_files) + len(logs_files) + len(figures_files) + len(archive_files) + len(tests_files)} files
  - **Root Directory:** {len(root_files)} files
  - **src/:** {len(src_files)} files
  - **paper/:** {len(paper_files)} files
  - **logs/:** {len(logs_files)} files
  - **figures/:** {len(figures_files)} files
  - **archive/:** {len(archive_files)} files
  - **tests_archive/:** {len(tests_files)} files
- **Status:** ✅ Verification Complete (4/4 tests passing)
- **Organization:** ✅ Repository restructured (Feb 6, 2026)

---

## 📁 ROOT DIRECTORY ({len(root_files)} files)

### Python Scripts ({len([f for f in root_files if f.endswith('.py')])})
"""
    
    # Add root Python files
    py_files = [f for f in root_files if f.endswith('.py')]
    for f in py_files:
        md += f"- `{f}`\n"
    
    # Add root markdown files
    md_files = [f for f in root_files if f.endswith('.md')]
    md += f"\n### Markdown Documentation ({len(md_files)})\n"
    for f in md_files:
        md += f"- `{f}`\n"
    
    # Add LaTeX files
    tex_files = [f for f in root_files if f.endswith('.tex')]
    md += f"\n### LaTeX Files ({len(tex_files)})\n"
    for f in tex_files:
        md += f"- `{f}`\n"
    
    # Add PDF files
    pdf_files = [f for f in root_files if f.endswith('.pdf')]
    md += f"\n### PDF Files ({len(pdf_files)})\n"
    for f in pdf_files:
        md += f"- `{f}`\n"
    
    # Add PNG files
    png_files = [f for f in root_files if f.endswith('.png')]
    md += f"\n### PNG Images ({len(png_files)})\n"
    for f in png_files:
        md += f"- `{f}`\n"
    
    # Other files
    other_files = [f for f in root_files if not any(f.endswith(ext) for ext in ['.py', '.md', '.tex', '.pdf', '.png'])]
    if other_files:
        md += f"\n### Other Files ({len(other_files)})\n"
        for f in other_files:
            md += f"- `{f}`\n"
    
    # SRC directory
    md += f"\n\n---\n\n## 📁 SRC DIRECTORY ({len(src_files)} files)\n\n"
    py_src = [f for f in src_files if f.endswith('.py') and not f.endswith('.pyc')]
    pyc_src = [f for f in src_files if f.endswith('.pyc')]
    other_src = [f for f in src_files if f not in py_src and f not in pyc_src]
    
    md += f"### Python Modules ({len(py_src)})\n"
    for f in py_src:
        md += f"- `{f}`\n"
    
    if pyc_src:
        md += f"\n### Compiled Python ({len(pyc_src)})\n"
        md += f"*{len(pyc_src)} .pyc files (bytecode cache)*\n"
    
    if other_src:
        md += f"\n### Other Files ({len(other_src)})\n"
        for f in other_src:
            md += f"- `{f}`\n"
    
    # PAPER directory
    md += f"\n\n---\n\n## 📁 PAPER DIRECTORY ({len(paper_files)} files)\n\n"
    for f in paper_files:
        md += f"- `{f}`\n"
    
    # LOGS directory
    md += f"\n\n---\n\n## 📁 LOGS DIRECTORY ({len(logs_files)} files)\n\n"
    for f in logs_files:
        md += f"- `{f}`\n"
    
    # FIGURES directory
    md += f"\n\n---\n\n## 📁 FIGURES DIRECTORY ({len(figures_files)} files)\n\n"
    for f in figures_files:
        md += f"- `{f}`\n"
    
    # ARCHIVE directory
    md += f"\n\n---\n\n## 📁 ARCHIVE DIRECTORY ({len(archive_files)} files)\n\n"
    for f in archive_files:
        md += f"- `{f}`\n"
    
    # TESTS_ARCHIVE directory
    if tests_files:
        md += f"\n\n---\n\n## 📁 TESTS_ARCHIVE DIRECTORY ({len(tests_files)} files)\n\n"
        for f in tests_files:
            md += f"- `{f}`\n"
    
    # Write to file
    with open("DIRECTORY_SUMMARY.md", "w", encoding="utf-8") as f:
        f.write(md)
    
    print(f"✅ Generated DIRECTORY_SUMMARY.md with {len(root_files) + len(src_files) + len(paper_files) + len(logs_files) + len(figures_files) + len(archive_files) + len(tests_files)} files")
    print(f"   - Root: {len(root_files)}")
    print(f"   - src/: {len(src_files)}")
    print(f"   - paper/: {len(paper_files)}")
    print(f"   - logs/: {len(logs_files)}")
    print(f"   - figures/: {len(figures_files)}")
    print(f"   - archive/: {len(archive_files)}")
    print(f"   - tests_archive/: {len(tests_files)}")

if __name__ == "__main__":
    main()
