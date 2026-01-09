"""
Compile LaTeX using LaTeX.Online API
Simple, no authentication required
"""

import requests
import time
import zipfile
import io
from pathlib import Path

def compile_latex_online(tex_file):
    """Compile LaTeX using latex.online service"""
    print("Compiling LaTeX document using latex.online API...")
    
    tex_path = Path(tex_file)
    if not tex_path.exists():
        print(f"Error: {tex_file} not found!")
        return False
    
    # Read the tex file
    with open(tex_path, 'r', encoding='utf-8') as f:
        tex_content = f.read()
    
    # Collect figure files
    figure_files = list(sorted(tex_path.parent.glob('Figure*.png')))
    
    print(f"Uploading {tex_path.name} and {len(figure_files)} figures...")
    
    try:
        # Method 1: Try latexonline.cc with correct URL
        url = f'https://latexonline.cc/compile?text={requests.utils.quote(tex_content)}'
        
        print("Attempting compilation via latexonline.cc...")
        response = requests.get(url, timeout=120)
        
        if response.status_code == 200 and response.headers.get('content-type') == 'application/pdf':
            # Save PDF
            pdf_path = tex_path.with_suffix('.pdf')
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
            print(f"\n✓ SUCCESS! PDF created: {pdf_path}")
            print(f"   Size: {len(response.content)} bytes")
            return True
        else:
            print(f"Method 1 failed: Status {response.status_code}")
            
    except Exception as e:
        print(f"Compilation error: {e}")
    
    print("\n" + "="*70)
    print("MANUAL COMPILATION REQUIRED")
    print("="*70)
    print("\nThe LaTeX file has been updated with the Code Availability section.")
    print("\nTo generate the PDF, please use one of these options:")
    print("\n1. OVERLEAF (Easiest - Free online):")
    print("   - Go to https://www.overleaf.com")
    print("   - Create free account / login")
    print("   - New Project → Upload Project")
    print("   - Upload manuscript.tex and all Figure*.png files")
    print("   - Click 'Recompile' → Download PDF")
    print("\n2. INSTALL LaTeX LOCALLY:")
    print("   - Windows: Download MiKTeX from https://miktex.org/download")
    print("   - After install, run: compile_manuscript.bat")
    print("\n3. TEXLIVE.NET (Online, no account):")
    print("   - Go to https://texlive.net")
    print("   - Paste manuscript.tex content")
    print("   - Upload figures as additional files")
    print("   - Click Typeset → Download PDF")
    
    return False

if __name__ == "__main__":
    compile_latex_online('manuscript.tex')
