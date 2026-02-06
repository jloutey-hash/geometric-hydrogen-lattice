"""
Alternative PDF Generation Script for Manuscript
Provides multiple options for creating a PDF from the manuscript

Requirements: Install as needed with:
    pip install reportlab pypandoc pillow
    
Or use the built-in weasyprint approach
"""

import os
import sys
import subprocess
from pathlib import Path

def check_latex_installed():
    """Check if pdflatex is installed"""
    try:
        result = subprocess.run(['pdflatex', '--version'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

def compile_with_latex(tex_file):
    """Compile LaTeX document to PDF"""
    print("Compiling with pdflatex...")
    tex_path = Path(tex_file)
    
    if not tex_path.exists():
        print(f"Error: {tex_file} not found!")
        return False
    
    # Change to directory containing tex file
    original_dir = os.getcwd()
    os.chdir(tex_path.parent)
    
    try:
        # Run pdflatex twice (for cross-references)
        for i in range(2):
            print(f"  Pass {i+1}/2...")
            result = subprocess.run(['pdflatex', '-interaction=nonstopmode', tex_path.name],
                                  capture_output=True,
                                  text=True,
                                  timeout=60)
            
            if result.returncode != 0:
                print(f"Error during compilation:")
                print(result.stdout[-1000:])  # Last 1000 chars
                return False
        
        pdf_file = tex_path.with_suffix('.pdf')
        if pdf_file.exists():
            print(f"\n✓ Success! PDF created: {pdf_file}")
            print(f"  Size: {pdf_file.stat().st_size / 1024:.1f} KB")
            return True
        else:
            print("Error: PDF file not created")
            return False
            
    finally:
        os.chdir(original_dir)

def try_pandoc_conversion(tex_file):
    """Try converting with pandoc"""
    print("\nTrying pandoc conversion...")
    tex_path = Path(tex_file)
    pdf_file = tex_path.with_suffix('.pdf')
    
    try:
        result = subprocess.run([
            'pandoc',
            str(tex_path),
            '-o', str(pdf_file),
            '--pdf-engine=xelatex',
            '-V', 'geometry:margin=1in'
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0 and pdf_file.exists():
            print(f"✓ Success with pandoc! PDF: {pdf_file}")
            return True
        else:
            print("Pandoc conversion failed:")
            print(result.stderr[:500])
            return False
            
    except FileNotFoundError:
        print("Pandoc not installed. Install from: https://pandoc.org")
        return False
    except subprocess.TimeoutExpired:
        print("Pandoc conversion timed out")
        return False

def create_simple_html_preview(tex_file):
    """Create a simple HTML preview (figures + text)"""
    print("\nCreating HTML preview...")
    tex_path = Path(tex_file)
    html_file = tex_path.with_suffix('.html')
    
    # Read manuscript text
    manuscript_txt = tex_path.parent / 'MANUSCRIPT.txt'
    if not manuscript_txt.exists():
        print(f"Error: {manuscript_txt} not found")
        return False
    
    with open(manuscript_txt, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all figures
    figures = sorted(tex_path.parent.glob('Figure*.png'))
    
    # Create HTML
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Exact Discretization of Quantum Angular Momentum</title>
    <style>
        body {{
            font-family: 'Times New Roman', serif;
            max-width: 800px;
            margin: 40px auto;
            padding: 20px;
            line-height: 1.6;
        }}
        h1 {{
            text-align: center;
            color: #333;
        }}
        .figure {{
            margin: 30px 0;
            text-align: center;
        }}
        .figure img {{
            max-width: 100%;
            border: 1px solid #ddd;
            padding: 10px;
            background: white;
        }}
        .caption {{
            font-style: italic;
            margin-top: 10px;
            text-align: left;
        }}
        pre {{
            white-space: pre-wrap;
            font-family: 'Courier New', monospace;
            background: #f5f5f5;
            padding: 15px;
            border-left: 3px solid #333;
        }}
    </style>
</head>
<body>
    <h1>Exact Discretization of Quantum Angular Momentum:<br>
    A Sparse-Matrix Construction Preserving SU(2) Commutation Relations</h1>
"""
    
    # Add figures
    for i, fig in enumerate(figures, 1):
        html_content += f"""
    <div class="figure">
        <img src="{fig.name}" alt="Figure {i}">
        <div class="caption"><strong>Figure {i}:</strong> {fig.stem.replace('_', ' ')}</div>
    </div>
"""
    
    # Add manuscript text
    html_content += f"""
    <h2>Manuscript Text</h2>
    <pre>{content}</pre>
</body>
</html>
"""
    
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✓ HTML preview created: {html_file}")
    print(f"  Open in browser and use Print → Save as PDF")
    return True

def print_instructions():
    """Print detailed instructions for PDF creation"""
    print("""
================================================================================
PDF CREATION OPTIONS FOR MANUSCRIPT
================================================================================

Your system analysis:
""")
    
    has_latex = check_latex_installed()
    print(f"  LaTeX (pdflatex): {'✓ Installed' if has_latex else '✗ Not installed'}")
    print(f"  Python: ✓ {sys.version.split()[0]}")
    
    print("""
================================================================================
RECOMMENDED APPROACH: Use Overleaf (easiest, no installation)
================================================================================

1. Go to https://www.overleaf.com (free account)
2. Create new project → Upload manuscript.tex
3. Upload all 6 Figure*.png files
4. Click "Recompile"
5. Download PDF

This is the FASTEST and most reliable method!

================================================================================
LOCAL COMPILATION OPTIONS:
================================================================================
""")
    
    if has_latex:
        print("Option 1: Compile LaTeX directly (RECOMMENDED - LaTeX installed)")
        print("  python generate_pdf.py --latex")
        print()
    else:
        print("Option 1: Install LaTeX first")
        print("  Windows: Download MiKTeX from https://miktex.org")
        print("  Then run: python generate_pdf.py --latex")
        print()
    
    print("Option 2: Use Pandoc (if installed)")
    print("  Install: https://pandoc.org/installing.html")
    print("  Then run: python generate_pdf.py --pandoc")
    print()
    
    print("Option 3: Create HTML preview (always works)")
    print("  python generate_pdf.py --html")
    print("  Then open in browser → Print → Save as PDF")
    print()
    
    print("""
================================================================================
For detailed instructions, see: COMPILATION_GUIDE.txt
================================================================================
""")

def main():
    """Main execution"""
    base_dir = Path(__file__).parent
    tex_file = base_dir / 'manuscript.tex'
    
    if len(sys.argv) < 2:
        print_instructions()
        return
    
    option = sys.argv[1].lower()
    
    if option == '--latex':
        if check_latex_installed():
            compile_with_latex(tex_file)
        else:
            print("Error: pdflatex not found. Install LaTeX first.")
            print("See COMPILATION_GUIDE.txt for instructions.")
    
    elif option == '--pandoc':
        try_pandoc_conversion(tex_file)
    
    elif option == '--html':
        create_simple_html_preview(tex_file)
        print("\nNext: Open the HTML file in your browser")
        print("      Press Ctrl+P (Cmd+P on Mac)")
        print("      Choose 'Save as PDF'")
    
    else:
        print(f"Unknown option: {option}")
        print_instructions()

if __name__ == '__main__':
    main()
