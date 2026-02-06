cd "c:\Users\jlout\OneDrive\Desktop\Model study\SU(2) model"
Remove-Item geometric_atom_v4.aux, geometric_atom_v4.log, geometric_atom_v4.out, geometric_atom_v4.pdf -ErrorAction SilentlyContinue
pdflatex -interaction=nonstopmode geometric_atom_v4.tex
bibtex geometric_atom_v4
pdflatex -interaction=nonstopmode geometric_atom_v4.tex
pdflatex -interaction=nonstopmode geometric_atom_v4.tex
Write-Host "Compilation complete. Check geometric_atom_v4.pdf"
