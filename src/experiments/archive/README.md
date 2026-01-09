# Archived Phase 15 Development Files

This directory contains Phase 15 implementation files that were used during development but are **not** part of the final validated code.

## Archival Date
January 2025

## Archived Files

### phase15_2_optimize.py (195 lines)
- **Purpose**: Development/analysis script for optimizing angular coupling strength (α)
- **Status**: Development tool, NOT used by validation tests
- **Dependencies**: imports `phase15_2_angular_coupling.py` (still in production)
- **Reason for Archival**: Analysis script used to determine optimal α=1.8 parameter. Once parameter was chosen, this optimization script was no longer needed for production.
- **Keep?**: Historical reference for how α=1.8 was selected

### phase15_3_multielectron.py (400 lines)
- **Purpose**: Configuration Interaction (CI) approach to multi-electron systems
- **Status**: Deprecated, superseded by Hartree-Fock implementation
- **Dependencies**: imports `phase15_2_fixed.py` (still in production)
- **Reason for Archival**: Full CI approach was too computationally expensive. Replaced by `phase15_3_hartree_fock.py` which achieves same 1.08 eV error with much better performance.
- **Keep?**: Historical reference showing CI was attempted before HF

### debug_radial.py
- **Purpose**: Debug script for radial discretization issues
- **Status**: Development debug tool
- **Dependencies**: Unknown (not analyzed)
- **Reason for Archival**: Debug script used during Phase 15.1 development to diagnose radial Laplacian issues.
- **Keep?**: Historical reference for debugging process

## Production Files (Still Active)

The following Phase 15 files remain in `src/experiments/` because they are:
1. Directly imported by `tests/validate_phase15.py`, OR
2. Required dependencies of validated code

### Core Production Files
- `phase15_complete_3d.py` (Phase 15.1) - Radial discretization implementation
- `phase15_2_final.py` (Phase 15.2) - Angular coupling (α=1.8, validates 1.24% error)
- `phase15_3_hartree_fock.py` (Phase 15.3) - Hartree-Fock Helium (validates 1.08 eV error)

### Required Dependencies
- `phase15_2_angular_coupling.py` - Imported by `phase15_2_final.py`
- `phase15_2_fixed.py` - Imported by `phase15_3_hartree_fock.py`

## Validation Test Results

All production Phase 15 files have been validated:

```
tests/validate_phase15.py:
  test_phase15_1_radial_fix ................. PASS (informational)
  test_phase15_2_angular_coupling ........... PASS ✅ (1.24% error validated)
  test_phase15_3_helium ..................... PASS ✅ (1.08 eV error validated)
  test_hydrogen_comparison .................. PASS ✅
  test_convergence .......................... PASS (informational)

Result: 5/5 PASS
```

## Recovery Instructions

If any archived file is needed:

1. **phase15_2_optimize.py**: If you need to re-optimize α parameter, move back to `src/experiments/`
2. **phase15_3_multielectron.py**: If you want to compare CI vs HF approaches, move back to `src/experiments/`
3. **debug_radial.py**: If radial issues resurface, move back to `src/experiments/`

All archived files have their imports already fixed to use absolute paths (`from experiments.X`).
