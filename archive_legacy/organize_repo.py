#!/usr/bin/env python3
"""
Repository Organization Script
===============================

Reorganizes the research codebase for public GitHub release.

Directory Structure:
--------------------
/src/               - Core physics code (cleaned and standardized)
/paper/             - LaTeX manuscripts and generated PDFs
/logs/              - Markdown logs and research notes
/figures/           - Generated plots (PNG/PDF)
/archive/           - Deprecated scripts (for reference)
/tests/             - Test and validation scripts

Critical Files (Renamed for Clarity):
--------------------------------------
model_alpha.py              <- hydrogen_u1_impedance.py (Alpha calculator)
model_spectral_audit.py     <- physics_spectral_audit.py (Dimensionless proof)
model_lattice.py            <- paraboloid_lattice_su11.py (Core lattice)
model_su3.py                <- su3_impedance_analysis.py (SU(3) extension)

Usage:
------
python organize_repo.py [--dry-run] [--execute]

--dry-run   : Show what would be moved without actually moving files
--execute   : Actually perform the reorganization (USE WITH CAUTION)
"""

import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# ANSI color codes for terminal output
class Color:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


class RepositoryOrganizer:
    """Organizes research repository for public release."""
    
    def __init__(self, base_path: str = "."):
        self.base = Path(base_path).resolve()
        self.dry_run = True
        
        # Define target directories
        self.dirs = {
            'src': self.base / 'src',
            'paper': self.base / 'paper',
            'logs': self.base / 'logs',
            'figures': self.base / 'figures',
            'archive': self.base / 'archive',
            'tests': self.base / 'tests_archive'
        }
        
        # Define critical file mappings (source -> destination)
        self.core_files = {
            # Core physics models
            'hydrogen_u1_impedance.py': 'src/model_alpha.py',
            'physics_spectral_audit.py': 'src/model_spectral_audit.py',
            'paraboloid_lattice_su11.py': 'src/model_lattice.py',
            'su3_impedance_analysis.py': 'src/model_su3.py',
            'geometric_impedance_interface.py': 'src/model_interface.py',
            
            # Supporting modules
            'physics_alpha_refinement.py': 'src/compute_alpha.py',
            'physics_lattice_statistics.py': 'src/compute_statistics.py',
            'generate_manuscript_figures.py': 'src/generate_figures.py',
            
            # LaTeX manuscripts
            'geometric_atom_symplectic_revision.tex': 'paper/paper_alpha.tex',
            'holographic_hydrogen_atom.tex': 'paper/paper_holography.tex',
            'geometric_atom_symplectic_revision.pdf': 'paper/paper_alpha.pdf',
            'holographic_hydrogen_atom.pdf': 'paper/paper_holography.pdf',
        }
        
        # Patterns for automatic categorization
        self.patterns = {
            'logs': ['*.md', '*_SUMMARY.md', '*_COMPLETE.md', '*_STATUS.md'],
            'figures': ['*.png', '*.pdf', '*.jpg'],
            'archive': ['test_*.py', 'debug_*.py', 'run_*.py', 'verify_*.py', 'demo.py'],
            'tests': ['validate_*.py', '*_test.py'],
        }
    
    def create_directories(self):
        """Create target directory structure."""
        print(f"\n{Color.HEADER}=== Creating Directory Structure ==={Color.ENDC}")
        for name, path in self.dirs.items():
            if not self.dry_run:
                path.mkdir(exist_ok=True)
            status = "✓" if path.exists() or not self.dry_run else "→"
            print(f"{status} {name:15s} : {path.relative_to(self.base)}")
    
    def move_core_files(self):
        """Move and rename core physics files."""
        print(f"\n{Color.HEADER}=== Moving Core Files ==={Color.ENDC}")
        for src_name, dst_rel in self.core_files.items():
            src = self.base / src_name
            dst = self.base / dst_rel
            
            if src.exists():
                if not self.dry_run:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    print(f"{Color.OKGREEN}✓{Color.ENDC} {src_name:45s} → {dst_rel}")
                else:
                    print(f"{Color.OKCYAN}→{Color.ENDC} {src_name:45s} → {dst_rel}")
            else:
                print(f"{Color.WARNING}✗{Color.ENDC} {src_name:45s} (not found)")
    
    def organize_by_pattern(self):
        """Organize remaining files by pattern matching."""
        print(f"\n{Color.HEADER}=== Organizing by Pattern ==={Color.ENDC}")
        
        for category, patterns in self.patterns.items():
            target_dir = self.dirs.get(category, self.base / category)
            files_moved = 0
            
            for pattern in patterns:
                for file in self.base.glob(pattern):
                    # Skip if already in target directory
                    if file.parent == target_dir:
                        continue
                    
                    # Skip if it's a core file (already handled)
                    if file.name in self.core_files:
                        continue
                    
                    # Skip directories
                    if file.is_dir():
                        continue
                    
                    dst = target_dir / file.name
                    if not self.dry_run:
                        target_dir.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(file, dst)
                    files_moved += 1
            
            if files_moved > 0:
                print(f"{Color.OKGREEN}✓{Color.ENDC} {category:15s} : {files_moved} files")
    
    def generate_report(self):
        """Generate organization report."""
        print(f"\n{Color.HEADER}=== Organization Summary ==={Color.ENDC}")
        
        # Count files in each directory
        for name, path in self.dirs.items():
            if path.exists():
                count = sum(1 for _ in path.glob('*') if _.is_file())
                print(f"  {name:15s} : {count:3d} files")
        
        # Core files status
        print(f"\n{Color.HEADER}Core Physics Files:{Color.ENDC}")
        critical = [
            'src/model_alpha.py',
            'src/model_spectral_audit.py', 
            'src/model_lattice.py'
        ]
        for file in critical:
            path = self.base / file
            status = "✓" if path.exists() else "✗"
            color = Color.OKGREEN if path.exists() else Color.FAIL
            print(f"  {color}{status}{Color.ENDC} {file}")
    
    def run(self, dry_run: bool = True):
        """Execute the reorganization."""
        self.dry_run = dry_run
        
        mode_str = "DRY RUN" if dry_run else "EXECUTION"
        print(f"\n{Color.BOLD}{'='*60}{Color.ENDC}")
        print(f"{Color.BOLD}Repository Organizer - {mode_str}{Color.ENDC}")
        print(f"{Color.BOLD}{'='*60}{Color.ENDC}")
        print(f"Base directory: {self.base}")
        
        if not dry_run:
            confirm = input(f"\n{Color.WARNING}⚠ This will modify files. Continue? (yes/no): {Color.ENDC}")
            if confirm.lower() != 'yes':
                print("Aborted.")
                return
        
        self.create_directories()
        self.move_core_files()
        self.organize_by_pattern()
        self.generate_report()
        
        if dry_run:
            print(f"\n{Color.OKCYAN}ℹ This was a dry run. Use --execute to apply changes.{Color.ENDC}")
        else:
            print(f"\n{Color.OKGREEN}✓ Reorganization complete!{Color.ENDC}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Organize research repository for GitHub release",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python organize_repo.py --dry-run     # Preview changes
  python organize_repo.py --execute     # Apply reorganization
        """
    )
    parser.add_argument('--dry-run', action='store_true', 
                        help='Show what would be moved without making changes')
    parser.add_argument('--execute', action='store_true',
                        help='Actually perform the reorganization')
    
    args = parser.parse_args()
    
    # Default to dry-run if no flags specified
    if not args.dry_run and not args.execute:
        args.dry_run = True
    
    organizer = RepositoryOrganizer()
    organizer.run(dry_run=args.dry_run)


if __name__ == '__main__':
    main()
