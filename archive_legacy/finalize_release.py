#!/usr/bin/env python3
"""
Release Finalization Script - "Gold Master" Preparation
========================================================
Performs scorched-earth cleanup for publication-ready repository.

Operations:
1. Archive all non-essential root files
2. Standardize the Paper Trilogy in paper/
3. Fix README.md links
4. Run smoke test verification
"""

import os
import shutil
import re
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Color codes for terminal output
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
CYAN = '\033[96m'
RESET = '\033[0m'
BOLD = '\033[1m'

def print_header(text):
    """Print a bold header."""
    print(f"\n{BOLD}{CYAN}{'='*70}{RESET}")
    print(f"{BOLD}{CYAN}{text.center(70)}{RESET}")
    print(f"{BOLD}{CYAN}{'='*70}{RESET}\n")

def print_success(text):
    """Print success message."""
    print(f"{GREEN}✓ {text}{RESET}")

def print_warning(text):
    """Print warning message."""
    print(f"{YELLOW}⚠ {text}{RESET}")

def print_error(text):
    """Print error message."""
    print(f"{RED}✗ {text}{RESET}")

def print_info(text):
    """Print info message."""
    print(f"{BLUE}→ {text}{RESET}")

def print_alert(text):
    """Print alert message."""
    print(f"\n{RED}{BOLD}{'!'*70}{RESET}")
    print(f"{RED}{BOLD}{text.center(70)}{RESET}")
    print(f"{RED}{BOLD}{'!'*70}{RESET}\n")


class ReleaseManager:
    """Manages the release finalization process."""
    
    def __init__(self, dry_run=False):
        self.root = Path(".")
        self.archive_root = self.root / "archive_legacy"
        self.paper_dir = self.root / "paper"
        self.dry_run = dry_run
        
        # Whitelist of allowed root items
        self.allowed_root = {
            'README.md',
            'LICENSE',
            '.gitignore',
            'requirements.txt',
            'run_reproduction.py',
            'finalize_release.py',
            # Critical dependencies for run_reproduction.py
            'hydrogen_u1_impedance.py',
            'paraboloid_lattice_su11.py',
            'physics_spectral_audit.py',
            'geometric_impedance_interface.py',
            # Directories
            'src',
            'paper',
            'figures',
            'logs',
            'archive_legacy',  # Will be created
            '.git',  # Keep git directory
            '__pycache__',  # Python cache
        }
        
        # Paper trilogy mapping: (search_names) -> final_name
        self.paper_trilogy = {
            'Paper_1_Spectrum.pdf': [
                'geometric_atom_submission.pdf',
                'Paper_1_Spectrum.pdf',
                'paper_1_spectrum.pdf',
            ],
            'Paper_2_Alpha.pdf': [
                'paper_alpha.pdf',
                'geometric_atom_symplectic_revision.pdf',
                'Paper_2_Alpha.pdf',
                'paper_2_alpha.pdf',
            ],
            'Paper_3_Holography.pdf': [
                'paper_holography.pdf',
                'Paper_3_Holography.pdf',
                'paper_3_holography.pdf',
                'holographic_hydrogen_atom.pdf',
            ],
        }
        
        self.moved_files = []
        self.moved_count = 0
    
    def create_archive_structure(self):
        """Create archive directory structure."""
        print_header("TASK 1: Root Cleanup")
        print_info("Creating archive directory structure...")
        
        if not self.dry_run:
            self.archive_root.mkdir(exist_ok=True)
            (self.archive_root / "paper_drafts").mkdir(exist_ok=True)
        
        print_success(f"Archive structure ready: {self.archive_root}")
    
    def cleanup_root(self):
        """Move all non-whitelisted files from root to archive."""
        print_info("Scanning root directory...")
        
        items_to_move = []
        for item in self.root.iterdir():
            if item.name not in self.allowed_root:
                items_to_move.append(item)
        
        print_info(f"Found {len(items_to_move)} items to archive")
        
        if len(items_to_move) == 0:
            print_success("Root is already clean!")
            return
        
        for item in items_to_move:
            dest = self.archive_root / item.name
            
            if self.dry_run:
                print(f"  [DRY-RUN] Would move: {item.name} -> archive_legacy/")
            else:
                try:
                    if dest.exists():
                        if dest.is_dir():
                            shutil.rmtree(dest)
                        else:
                            dest.unlink()
                    
                    shutil.move(str(item), str(dest))
                    print(f"  Moved: {item.name}")
                    self.moved_count += 1
                except Exception as e:
                    print_error(f"Failed to move {item.name}: {e}")
        
        print_success(f"Archived {self.moved_count} items to archive_legacy/")
    
    def find_paper(self, search_names):
        """Find a paper by searching for possible names."""
        # Search in root
        for name in search_names:
            path = self.root / name
            if path.exists():
                return path
        
        # Search in paper/
        if self.paper_dir.exists():
            for name in search_names:
                path = self.paper_dir / name
                if path.exists():
                    return path
        
        return None
    
    def standardize_papers(self):
        """Curate paper directory to contain only the trilogy."""
        print_header("TASK 2: Paper Standardization")
        
        if not self.paper_dir.exists():
            if not self.dry_run:
                self.paper_dir.mkdir(exist_ok=True)
            print_info("Created paper/ directory")
        
        # Find and rename trilogy papers
        trilogy_found = {}
        for final_name, search_names in self.paper_trilogy.items():
            paper = self.find_paper(search_names)
            if paper:
                trilogy_found[final_name] = paper
                print_success(f"Found {final_name}: {paper}")
            else:
                print_warning(f"Missing {final_name} (searched: {', '.join(search_names)})")
        
        # Move trilogy to paper/ with standard names
        for final_name, source_path in trilogy_found.items():
            dest_path = self.paper_dir / final_name
            
            if self.dry_run:
                print(f"  [DRY-RUN] Would rename: {source_path} -> {dest_path}")
            else:
                try:
                    if source_path != dest_path:
                        if dest_path.exists():
                            dest_path.unlink()
                        shutil.copy2(str(source_path), str(dest_path))
                        print_info(f"Standardized: {final_name}")
                except Exception as e:
                    print_error(f"Failed to copy {source_path}: {e}")
        
        # Archive all other paper/ files
        if self.paper_dir.exists():
            archive_paper = self.archive_root / "paper_drafts"
            
            other_files = []
            for item in self.paper_dir.iterdir():
                if item.name not in self.paper_trilogy.keys():
                    other_files.append(item)
            
            if other_files:
                print_info(f"Archiving {len(other_files)} draft files...")
                
                for item in other_files:
                    dest = archive_paper / item.name
                    
                    if self.dry_run:
                        print(f"  [DRY-RUN] Would archive: {item.name}")
                    else:
                        try:
                            if dest.exists():
                                dest.unlink()
                            shutil.move(str(item), str(dest))
                            print(f"  Archived: {item.name}")
                        except Exception as e:
                            print_error(f"Failed to archive {item.name}: {e}")
        
        print_success("Paper trilogy curated successfully!")
    
    def fix_readme_links(self):
        """Update README.md to use standardized paper names."""
        print_header("TASK 3: README Link Verification")
        
        readme = self.root / "README.md"
        if not readme.exists():
            print_warning("README.md not found - skipping link fixes")
            return
        
        print_info("Reading README.md...")
        content = readme.read_text(encoding='utf-8')
        original_content = content
        
        # Define link mappings
        link_fixes = [
            # Volume I
            (r'geometric_atom_submission\.pdf', 'paper/Paper_1_Spectrum.pdf'),
            (r'paper_1_spectrum\.pdf', 'paper/Paper_1_Spectrum.pdf'),
            
            # Volume II
            (r'paper_alpha\.pdf', 'paper/Paper_2_Alpha.pdf'),
            (r'geometric_atom_symplectic_revision\.pdf', 'paper/Paper_2_Alpha.pdf'),
            (r'paper_2_alpha\.pdf', 'paper/Paper_2_Alpha.pdf'),
            
            # Volume III
            (r'paper_holography\.pdf', 'paper/Paper_3_Holography.pdf'),
            (r'holographic_hydrogen_atom\.pdf', 'paper/Paper_3_Holography.pdf'),
            (r'paper_3_holography\.pdf', 'paper/Paper_3_Holography.pdf'),
        ]
        
        fixes_applied = 0
        for pattern, replacement in link_fixes:
            if re.search(pattern, content, re.IGNORECASE):
                content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
                fixes_applied += 1
        
        # Ensure volume numbering is consistent
        volume_fixes = [
            (r'Volume\s+1\b', 'Volume I'),
            (r'Volume\s+2\b', 'Volume II'),
            (r'Volume\s+3\b', 'Volume III'),
            (r'Vol\.\s*1\b', 'Volume I'),
            (r'Vol\.\s*2\b', 'Volume II'),
            (r'Vol\.\s*3\b', 'Volume III'),
        ]
        
        for pattern, replacement in volume_fixes:
            content = re.sub(pattern, replacement, content)
        
        if content != original_content:
            if self.dry_run:
                print(f"  [DRY-RUN] Would apply {fixes_applied} link fixes to README.md")
            else:
                readme.write_text(content, encoding='utf-8')
                print_success(f"Applied {fixes_applied} link fixes to README.md")
        else:
            print_success("README.md links are already correct!")
    
    def run_smoke_test(self):
        """Run verification script to ensure nothing broke."""
        print_header("TASK 4: Smoke Test Verification")
        
        if self.dry_run:
            print_info("[DRY-RUN] Would run: python run_reproduction.py")
            return
        
        print_info("Running verification suite...")
        print_info("Command: python run_reproduction.py\n")
        
        try:
            result = subprocess.run(
                [sys.executable, "run_reproduction.py"],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                print(result.stdout)
                print_success("SMOKE TEST PASSED!")
                print_success("All imports and calculations verified ✓")
            else:
                print(result.stdout)
                print(result.stderr)
                print_alert("RED ALERT: SMOKE TEST FAILED!")
                print_error("The verification script failed!")
                print_error("Imports or calculations may be broken.")
                print_error("Check the output above for details.")
                return False
        
        except subprocess.TimeoutExpired:
            print_alert("RED ALERT: SMOKE TEST TIMEOUT!")
            print_error("Verification took longer than 5 minutes.")
            return False
        
        except Exception as e:
            print_alert("RED ALERT: SMOKE TEST ERROR!")
            print_error(f"Failed to run verification: {e}")
            return False
        
        return True
    
    def generate_report(self):
        """Generate final release report."""
        print_header("RELEASE FINALIZATION SUMMARY")
        
        print(f"{BOLD}Repository State:{RESET}")
        print(f"  Root files: {len(list(self.root.iterdir()))} items")
        print(f"  Archived: {self.moved_count} items -> archive_legacy/")
        
        if self.paper_dir.exists():
            paper_count = len(list(self.paper_dir.glob("*.pdf")))
            print(f"  Papers: {paper_count} PDFs in paper/")
        
        print(f"\n{BOLD}Trilogy Status:{RESET}")
        for paper_name in self.paper_trilogy.keys():
            paper_path = self.paper_dir / paper_name
            if paper_path.exists():
                size = paper_path.stat().st_size / 1024  # KB
                print_success(f"{paper_name} ({size:.1f} KB)")
            else:
                print_warning(f"{paper_name} - MISSING!")
        
        print(f"\n{BOLD}{GREEN}✓ GOLD MASTER RELEASE READY{RESET}")
        print(f"{CYAN}Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{RESET}\n")


def main():
    """Main execution function."""
    print(f"\n{BOLD}{MAGENTA}{'='*70}{RESET}")
    print(f"{BOLD}{MAGENTA}{'RELEASE FINALIZATION SCRIPT'.center(70)}{RESET}")
    print(f"{BOLD}{MAGENTA}{'Gold Master Preparation'.center(70)}{RESET}")
    print(f"{BOLD}{MAGENTA}{'='*70}{RESET}\n")
    
    # Parse command line arguments
    dry_run = '--dry-run' in sys.argv or '-n' in sys.argv
    
    if dry_run:
        print_warning("DRY-RUN MODE: No changes will be made\n")
    else:
        print_warning("This will reorganize your repository!")
        response = input(f"{YELLOW}Continue? (yes/no): {RESET}").strip().lower()
        if response != 'yes':
            print_error("Aborted by user")
            return 1
    
    # Execute release process
    manager = ReleaseManager(dry_run=dry_run)
    
    try:
        manager.create_archive_structure()
        manager.cleanup_root()
        manager.standardize_papers()
        manager.fix_readme_links()
        
        if not dry_run:
            success = manager.run_smoke_test()
            if not success:
                print_error("\n⚠️  Release verification failed!")
                print_error("Repository has been reorganized but verification did not pass.")
                return 1
        
        manager.generate_report()
        
        if dry_run:
            print(f"\n{CYAN}To execute: python finalize_release.py{RESET}")
        else:
            print(f"\n{GREEN}{BOLD}🎉 REPOSITORY READY FOR PUBLICATION! 🎉{RESET}\n")
        
        return 0
    
    except KeyboardInterrupt:
        print_error("\n\nInterrupted by user")
        return 1
    except Exception as e:
        print_alert("FATAL ERROR")
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
