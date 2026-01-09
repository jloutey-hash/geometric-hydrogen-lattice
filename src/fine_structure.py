"""
Fine structure constant exploration from lattice geometry.

This module investigates 10 different geometric approaches to derive
the fine structure constant α ≈ 1/137.036 from the discrete polar lattice.
"""

import numpy as np
from scipy import sparse
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt


# Fine structure constant (2018 CODATA value)
ALPHA_FINE = 1.0 / 137.035999084


class FineStructureExplorer:
    """
    Explore geometric origins of fine structure constant α from lattice.
    
    This class implements 10 research tracks investigating whether α
    emerges naturally from the discrete polar lattice geometry.
    """
    
    def __init__(self, lattice, operators=None, angular_momentum=None, spin_ops=None):
        """
        Initialize fine structure explorer.
        
        Parameters
        ----------
        lattice : PolarLattice
            The discrete polar lattice
        operators : LatticeOperators, optional
            Lattice differential operators
        angular_momentum : AngularMomentumOperators, optional
            Angular momentum operators
        spin_ops : SpinOperators, optional
            Spin operators
        """
        self.lattice = lattice
        self.operators = operators
        self.angular_momentum = angular_momentum
        self.spin_ops = spin_ops
        
        self.results = {}  # Store all findings
        
    def explore_all(self, verbose=True):
        """
        Run all 10 exploration tracks.
        
        Parameters
        ----------
        verbose : bool
            Print progress messages
            
        Returns
        -------
        dict
            Results from all tracks
        """
        tracks = [
            ('Track 1: Geometric Phase', self.track1_geometric_phase),
            ('Track 2: Shell Ratios', self.track2_shell_ratios),
            ('Track 3: L² Corrections', self.track3_L2_corrections),
            ('Track 4: Overlap Analysis', self.track4_overlap_analysis),
            ('Track 5: Coupling Constants', self.track5_coupling_constants),
            ('Track 6: Spin-Orbit', self.track6_spin_orbit),
            ('Track 7: Recursion Relations', self.track7_recursion),
            ('Track 8: Gauge Theory', self.track8_gauge_theory),
            ('Track 9: Information Theory', self.track9_information),
            ('Track 10: Asymptotic Expansion', self.track10_asymptotic),
        ]
        
        for name, track_func in tracks:
            if verbose:
                print(f"\n{'='*70}")
                print(f"{name}")
                print('='*70)
            
            result = track_func(verbose=verbose)
            self.results[name] = result
            
            if verbose:
                self._print_track_summary(name, result)
        
        # Final synthesis
        if verbose:
            print(f"\n{'='*70}")
            print("SYNTHESIS")
            print('='*70)
        
        synthesis = self.synthesize_results()
        self.results['Synthesis'] = synthesis
        
        if verbose:
            self._print_synthesis(synthesis)
        
        return self.results
    
    def track1_geometric_phase(self, verbose=False):
        """
        Track 1: Geometric Phase and Berry Curvature
        
        Investigate Berry phase around lattice rings and hemisphere transport.
        """
        if verbose:
            print("\n[1.1] Computing Berry connection around ℓ-rings...")
        
        results = {
            'track': 1,
            'name': 'Geometric Phase and Berry Curvature',
            'candidates': []
        }
        
        # For each ℓ, compute geometric phase around ring
        for ℓ in range(min(6, self.lattice.ℓ_max + 1)):
            N_ℓ = 2 * (2 * ℓ + 1)
            
            # Angular spacing
            dtheta = 2 * np.pi / N_ℓ
            
            # Geometric phase for one revolution
            # For angular momentum eigenstates, phase = 2π*m_ℓ
            # Look for corrections
            
            if ℓ > 0:
                # Average magnetic quantum number spacing
                m_spacing = 1.0  # Δm_ℓ = 1 between adjacent orbitals
                
                # Phase per step
                phase_per_step = 2 * np.pi / (2 * ℓ + 1)  # For orbital
                spin_phase_per_step = 2 * np.pi / 2  # For spin flip
                
                # Total phase for full revolution (both orbital and spin)
                total_phase = 2 * np.pi * ℓ  # From orbital
                
                # Correction from discrete sampling
                correction = abs(phase_per_step - (2*np.pi*m_spacing)/(2*ℓ+1))
                
                # Dimensionless ratio
                ratio = correction / (2 * np.pi)
                
                results['candidates'].append({
                    'ℓ': ℓ,
                    'description': f'Phase correction ratio (ℓ={ℓ})',
                    'value': ratio,
                    'deviation_from_alpha': abs(ratio - ALPHA_FINE),
                    'relative_error': abs(ratio - ALPHA_FINE) / ALPHA_FINE
                })
        
        # Hemisphere separation phase
        if verbose:
            print("[1.2] Computing hemisphere transport phase...")
        
        # Phase for transporting from north to south pole
        # In continuous case: geometric phase = solid angle = 2π(1-cos(θ))
        # For full hemisphere: 2π
        
        # Discrete case: sum over latitude bands
        n_bands = self.lattice.ℓ_max + 1
        discrete_solid_angle = 0
        
        for ℓ in range(self.lattice.ℓ_max + 1):
            # Each band subtends solid angle
            theta_ℓ = np.pi * (ℓ + 0.5) / (self.lattice.ℓ_max + 1)
            dtheta_band = np.pi / (self.lattice.ℓ_max + 1)
            
            # Approximate solid angle
            dOmega = 2 * np.pi * np.sin(theta_ℓ) * dtheta_band
            discrete_solid_angle += dOmega
        
        # Ratio of discrete to continuous
        continuous_solid_angle = 2 * np.pi  # Hemisphere
        solid_angle_ratio = discrete_solid_angle / continuous_solid_angle
        solid_angle_deficit = abs(1 - solid_angle_ratio)
        
        results['candidates'].append({
            'description': 'Solid angle deficit ratio',
            'value': solid_angle_deficit,
            'deviation_from_alpha': abs(solid_angle_deficit - ALPHA_FINE),
            'relative_error': abs(solid_angle_deficit - ALPHA_FINE) / ALPHA_FINE
        })
        
        # Berry curvature from spin texture
        if verbose:
            print("[1.3] Analyzing spin texture Berry curvature...")
        
        # For spin-1/2 on sphere, Berry curvature = ±1/2 (magnetic monopole)
        # Integrated Berry phase = ±2π for full sphere
        # Check if discretization introduces corrections ~ α
        
        berry_phase_continuous = 2 * np.pi  # Spin-1/2 monopole
        # Discrete approximation (from lattice point density)
        n_points_total = len(self.lattice.points)
        expected_points_continuous = 4 * np.pi * (self.lattice.ℓ_max + 1)**2  # Rough estimate
        
        discretization_factor = n_points_total / expected_points_continuous
        berry_correction = abs(1 - discretization_factor)
        
        results['candidates'].append({
            'description': 'Berry phase discretization correction',
            'value': berry_correction,
            'deviation_from_alpha': abs(berry_correction - ALPHA_FINE),
            'relative_error': abs(berry_correction - ALPHA_FINE) / ALPHA_FINE
        })
        
        return results
    
    def track2_shell_ratios(self, verbose=False):
        """
        Track 2: Shell Closure Ratios and Magic Numbers
        
        Analyze ratios of magic numbers (2, 8, 18, 32) for α.
        """
        if verbose:
            print("\n[2.1] Computing shell closure ratios...")
        
        results = {
            'track': 2,
            'name': 'Shell Closure Ratios',
            'candidates': []
        }
        
        # Magic numbers
        magic = [2, 8, 18, 32, 50, 72, 98, 128][:self.lattice.n_max]
        
        # Successive ratios
        for i in range(len(magic) - 1):
            ratio = magic[i] / magic[i+1]
            results['candidates'].append({
                'description': f'N({i+1})/N({i+2}) = {magic[i]}/{magic[i+1]}',
                'value': ratio,
                'deviation_from_alpha': abs(ratio - ALPHA_FINE),
                'relative_error': abs(ratio - ALPHA_FINE) / ALPHA_FINE
            })
            
            # Inverse ratio
            inv_ratio = magic[i+1] / magic[i]
            results['candidates'].append({
                'description': f'N({i+2})/N({i+1}) = {magic[i+1]}/{magic[i]}',
                'value': inv_ratio,
                'deviation_from_alpha': abs(inv_ratio - ALPHA_FINE),
                'relative_error': abs(inv_ratio - ALPHA_FINE) / ALPHA_FINE
            })
        
        # Difference ratios
        for i in range(len(magic) - 1):
            diff_ratio = (magic[i+1] - magic[i]) / magic[i+1]
            results['candidates'].append({
                'description': f'ΔN/N = ({magic[i+1]}-{magic[i]})/{magic[i+1]}',
                'value': diff_ratio,
                'deviation_from_alpha': abs(diff_ratio - ALPHA_FINE),
                'relative_error': abs(diff_ratio - ALPHA_FINE) / ALPHA_FINE
            })
        
        # Cumulative fractions
        if verbose:
            print("[2.2] Computing cumulative filling fractions...")
        
        for i in range(1, min(4, len(magic))):
            for j in range(i+1, min(5, len(magic))):
                frac = magic[i] / (magic[i] + magic[j])
                results['candidates'].append({
                    'description': f'Cumulative: N{i+1}/(N{i+1}+N{j+1}) = {magic[i]}/({magic[i]}+{magic[j]})',
                    'value': frac,
                    'deviation_from_alpha': abs(frac - ALPHA_FINE),
                    'relative_error': abs(frac - ALPHA_FINE) / ALPHA_FINE
                })
        
        # Angular momentum sums
        if verbose:
            print("[2.3] Analyzing angular momentum sum rules...")
        
        for n in range(1, min(6, self.lattice.n_max + 1)):
            # Σ(2ℓ+1) = n²
            sum_2l_plus_1 = n**2
            
            # Σ√(ℓ(ℓ+1))
            sum_sqrt_l = sum(np.sqrt(ℓ*(ℓ+1)) for ℓ in range(n))
            
            # Ratio
            ratio = sum_sqrt_l / sum_2l_plus_1
            results['candidates'].append({
                'description': f'Σ√(ℓ(ℓ+1))/n² for n={n}',
                'value': ratio,
                'deviation_from_alpha': abs(ratio - ALPHA_FINE),
                'relative_error': abs(ratio - ALPHA_FINE) / ALPHA_FINE
            })
        
        return results
    
    def track3_L2_corrections(self, verbose=False):
        """
        Track 3: L² Eigenvalue Structure and Quantum Corrections
        
        Look for α in zero-point energies and quantum corrections.
        """
        if verbose:
            print("\n[3.1] Computing zero-point energies...")
        
        results = {
            'track': 3,
            'name': 'L² Quantum Corrections',
            'candidates': []
        }
        
        if self.angular_momentum is None:
            if verbose:
                print("  [WARNING] Angular momentum operators not provided")
            return results
        
        # Build L² operator
        L2 = self.angular_momentum.build_L_squared()
        L2_eigenvalues = np.linalg.eigvalsh(L2.toarray())
        
        # Zero-point energy
        E_zp = np.sum(np.sqrt(np.abs(L2_eigenvalues)))
        
        # Classical energy (sum of ℓ(ℓ+1))
        E_classical = 0
        for ℓ in range(self.lattice.ℓ_max + 1):
            E_classical += (2*(2*ℓ+1)) * ℓ * (ℓ + 1)
        
        # Ratio
        ratio = E_zp / E_classical if E_classical > 0 else 0
        results['candidates'].append({
            'description': 'Zero-point / Classical energy ratio',
            'value': ratio,
            'deviation_from_alpha': abs(ratio - ALPHA_FINE),
            'relative_error': abs(ratio - ALPHA_FINE) / ALPHA_FINE if ALPHA_FINE > 0 else np.inf
        })
        
        # Quantum corrections as series
        if verbose:
            print("[3.2] Analyzing quantum correction series...")
        
        # For each ℓ, compute correction to ℓ(ℓ+1)
        for ℓ in range(1, min(6, self.lattice.ℓ_max + 1)):
            # Get eigenvalues for this ℓ
            mask = np.array([p['ℓ'] == ℓ for p in self.lattice.points])
            
            if np.any(mask):
                # Expected: ℓ(ℓ+1)
                expected = ℓ * (ℓ + 1)
                
                # Compute average correction (should be zero, but check fluctuations)
                # This is a placeholder - real quantum corrections require perturbation theory
                correction = 1.0 / (2 * ℓ + 1)  # Hypothetical 1/degeneracy correction
                
                results['candidates'].append({
                    'description': f'Hypothetical quantum correction 1/(2ℓ+1) for ℓ={ℓ}',
                    'value': correction,
                    'deviation_from_alpha': abs(correction - ALPHA_FINE),
                    'relative_error': abs(correction - ALPHA_FINE) / ALPHA_FINE
                })
        
        # G-factor style correction: g = 2(1 + α/2π)
        if verbose:
            print("[3.3] Testing g-factor analogy...")
        
        # If lattice had g-factor like correction, what would α be?
        # This is speculative - look for natural corrections in the structure
        
        return results
    
    def track4_overlap_analysis(self, verbose=False):
        """
        Track 4: Overlap Integrals and Wavefunction Normalization
        
        Analyze the 82% overlap efficiency with spherical harmonics.
        """
        if verbose:
            print("\n[4.1] Analyzing overlap efficiency...")
        
        results = {
            'track': 4,
            'name': 'Overlap Analysis',
            'candidates': []
        }
        
        # Known from Phase 4: average overlap η ≈ 0.82
        eta = 0.82
        
        # Various ratios
        ratios = {
            '(1-η)/η': (1 - eta) / eta,
            'η/(1-η)': eta / (1 - eta),
            '√(1-η²)': np.sqrt(1 - eta**2),
            '1-η': 1 - eta,
            '(1-η)/2': (1 - eta) / 2,
            'η²': eta**2,
            '1-η²': 1 - eta**2,
            '√(1-η)': np.sqrt(1 - eta),
        }
        
        for desc, value in ratios.items():
            results['candidates'].append({
                'description': f'Overlap ratio: {desc} with η=0.82',
                'value': value,
                'deviation_from_alpha': abs(value - ALPHA_FINE),
                'relative_error': abs(value - ALPHA_FINE) / ALPHA_FINE
            })
        
        # Selection rule compliance: 31%
        if verbose:
            print("[4.2] Analyzing selection rule violations...")
        
        compliance = 0.31
        violation = 1 - compliance
        
        violation_ratios = {
            'compliance': compliance,
            'violation': violation,
            'compliance/violation': compliance / violation,
            'violation/compliance': violation / compliance,
            '√(compliance)': np.sqrt(compliance),
            '√(violation)': np.sqrt(violation),
        }
        
        for desc, value in violation_ratios.items():
            results['candidates'].append({
                'description': f'Selection rules: {desc}',
                'value': value,
                'deviation_from_alpha': abs(value - ALPHA_FINE),
                'relative_error': abs(value - ALPHA_FINE) / ALPHA_FINE
            })
        
        # Solid angle ratios
        if verbose:
            print("[4.3] Computing discrete solid angle elements...")
        
        for ℓ in range(1, min(6, self.lattice.ℓ_max + 1)):
            N_ℓ = 2 * (2 * ℓ + 1)
            
            # Discrete solid angle per point (approximate)
            dOmega_discrete = 4 * np.pi / N_ℓ
            
            # For continuous sphere at this latitude
            theta_ℓ = np.pi * (ℓ + 0.5) / (self.lattice.ℓ_max + 1)
            dtheta = np.pi / (self.lattice.ℓ_max + 1)
            dphi = 2 * np.pi / (2 * ℓ + 1)
            
            dOmega_continuous = np.sin(theta_ℓ) * dtheta * dphi
            
            ratio = dOmega_discrete / dOmega_continuous if dOmega_continuous > 0 else 0
            
            results['candidates'].append({
                'description': f'Solid angle ratio for ℓ={ℓ}',
                'value': ratio,
                'deviation_from_alpha': abs(ratio - ALPHA_FINE),
                'relative_error': abs(ratio - ALPHA_FINE) / ALPHA_FINE if ALPHA_FINE > 0 else np.inf
            })
        
        return results
    
    def track5_coupling_constants(self, verbose=False):
        """
        Track 5: Radial-Angular Coupling Constants
        
        This would require implementing varying coupling and optimization.
        Placeholder for now.
        """
        if verbose:
            print("\n[5.1] Analyzing radial-angular coupling...")
            print("  [TODO] Requires Hamiltonian with variable coupling")
        
        results = {
            'track': 5,
            'name': 'Radial-Angular Coupling',
            'candidates': [],
            'status': 'TODO - requires variable Hamiltonian'
        }
        
        # Placeholder: from Phase 6, we know fitted A = -2.13 vs theoretical -13.6
        A_fitted = -2.13
        A_hydrogen = -13.6
        
        ratio = abs(A_fitted / A_hydrogen)
        results['candidates'].append({
            'description': 'Energy scale ratio |A_fitted/A_hydrogen|',
            'value': ratio,
            'deviation_from_alpha': abs(ratio - ALPHA_FINE),
            'relative_error': abs(ratio - ALPHA_FINE) / ALPHA_FINE
        })
        
        return results
    
    def track6_spin_orbit(self, verbose=False):
        """
        Track 6: Spin-Orbit Fine Structure Splitting
        
        Most direct physical connection to α.
        """
        if verbose:
            print("\n[6.1] Computing geometric spin-orbit coupling...")
        
        results = {
            'track': 6,
            'name': 'Spin-Orbit Fine Structure',
            'candidates': []
        }
        
        # Geometric λ: ratio of hemisphere separation to ring spacing
        for ℓ in range(1, min(6, self.lattice.ℓ_max + 1)):
            # Ring radius
            r_ℓ = 1 + 2 * ℓ
            
            # In 3D spherical lift, hemisphere separation is z-coordinate
            theta_north = np.pi * (ℓ + 0.5) / (self.lattice.ℓ_max + 1)
            theta_south = np.pi - theta_north
            
            z_north = np.cos(theta_north)
            z_south = np.cos(theta_south)
            
            hemisphere_sep = abs(z_north - z_south)
            
            # Ring spacing in 2D
            ring_spacing = 2  # r_{ℓ+1} - r_ℓ = 2
            
            lambda_geom = hemisphere_sep / ring_spacing
            
            results['candidates'].append({
                'description': f'Geometric λ = z_sep/r_spacing for ℓ={ℓ}',
                'value': lambda_geom,
                'deviation_from_alpha': abs(lambda_geom - ALPHA_FINE),
                'relative_error': abs(lambda_geom - ALPHA_FINE) / ALPHA_FINE
            })
            
            # Also try λ² (since fine structure goes as α²)
            lambda_sq = lambda_geom**2
            results['candidates'].append({
                'description': f'Geometric λ² for ℓ={ℓ}',
                'value': lambda_sq,
                'deviation_from_alpha': abs(lambda_sq - ALPHA_FINE),
                'relative_error': abs(lambda_sq - ALPHA_FINE) / ALPHA_FINE
            })
        
        # Alternative: angular momentum quantization ratio
        if verbose:
            print("[6.2] Testing spin-angular momentum ratios...")
        
        # Spin-orbit: <L·S> ~ ℓ/2 for j=ℓ+1/2, -（ℓ+1)/2 for j=ℓ-1/2
        # Energy splitting ∝ α²
        
        for ℓ in range(1, min(5, self.lattice.ℓ_max + 1)):
            # Ratio of splitting to total
            splitting_ratio = 1 / (2 * ℓ + 1)
            
            results['candidates'].append({
                'description': f'Splitting ratio 1/(2ℓ+1) for ℓ={ℓ}',
                'value': splitting_ratio,
                'deviation_from_alpha': abs(splitting_ratio - ALPHA_FINE),
                'relative_error': abs(splitting_ratio - ALPHA_FINE) / ALPHA_FINE
            })
        
        return results
    
    def track7_recursion(self, verbose=False):
        """
        Track 7: Fibonacci-like Recursion Relations
        
        Look for α in recursive patterns.
        """
        if verbose:
            print("\n[7.1] Analyzing recursion relations...")
        
        results = {
            'track': 7,
            'name': 'Recursion Relations',
            'candidates': []
        }
        
        # N_ℓ sequence: 2, 6, 10, 14, 18, ...
        N_sequence = [2 * (2 * ℓ + 1) for ℓ in range(min(10, self.lattice.ℓ_max + 1))]
        
        # Ratios N_ℓ/N_{ℓ-1}
        for i in range(1, len(N_sequence)):
            ratio = N_sequence[i-1] / N_sequence[i]
            results['candidates'].append({
                'description': f'N_{i-1}/N_{i} = {N_sequence[i-1]}/{N_sequence[i]}',
                'value': ratio,
                'deviation_from_alpha': abs(ratio - ALPHA_FINE),
                'relative_error': abs(ratio - ALPHA_FINE) / ALPHA_FINE
            })
        
        # Second differences
        if verbose:
            print("[7.2] Computing second-order differences...")
        
        for i in range(2, len(N_sequence)):
            delta1 = N_sequence[i] - N_sequence[i-1]
            delta2 = N_sequence[i-1] - N_sequence[i-2]
            
            if delta1 > 0:
                ratio = delta2 / delta1
                results['candidates'].append({
                    'description': f'Δ²N ratio at ℓ={i}',
                    'value': ratio,
                    'deviation_from_alpha': abs(ratio - ALPHA_FINE),
                    'relative_error': abs(ratio - ALPHA_FINE) / ALPHA_FINE
                })
        
        # Golden ratio connection
        if verbose:
            print("[7.3] Testing golden ratio relationships...")
        
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        
        golden_ratios = {
            '1/φ': 1/phi,
            '1/φ²': 1/phi**2,
            'φ-1': phi - 1,
            '2-φ': 2 - phi,
            '√5-2': np.sqrt(5) - 2,
            '(√5-2)/φ': (np.sqrt(5) - 2) / phi,
        }
        
        for desc, value in golden_ratios.items():
            results['candidates'].append({
                'description': f'Golden ratio: {desc}',
                'value': value,
                'deviation_from_alpha': abs(value - ALPHA_FINE),
                'relative_error': abs(value - ALPHA_FINE) / ALPHA_FINE
            })
        
        return results
    
    def track8_gauge_theory(self, verbose=False):
        """
        Track 8: Discrete Electromagnetism and Gauge Theory
        
        Placeholder - requires full gauge field implementation.
        """
        if verbose:
            print("\n[8.1] Gauge theory analysis...")
            print("  [TODO] Requires discrete gauge field implementation")
        
        results = {
            'track': 8,
            'name': 'Gauge Theory',
            'candidates': [],
            'status': 'TODO - requires gauge field implementation'
        }
        
        # Placeholder: Wilson loop would go around smallest ring
        # Minimal phase would be 2π/N_0 = 2π/2 = π
        
        return results
    
    def track9_information(self, verbose=False):
        """
        Track 9: Information-Theoretic Approach
        
        Compute entropy measures.
        """
        if verbose:
            print("\n[9.1] Computing entropy measures...")
        
        results = {
            'track': 9,
            'name': 'Information Theory',
            'candidates': []
        }
        
        # Uniform distribution entropy
        for n in range(1, min(6, self.lattice.n_max + 1)):
            N_shell = 2 * n**2
            
            # Shannon entropy for uniform distribution
            S = np.log(N_shell)
            
            # Normalize by log(N_max)
            N_max = 2 * self.lattice.n_max**2
            S_normalized = S / np.log(N_max) if N_max > 1 else 0
            
            results['candidates'].append({
                'description': f'Normalized entropy S/S_max for n={n}',
                'value': S_normalized,
                'deviation_from_alpha': abs(S_normalized - ALPHA_FINE),
                'relative_error': abs(S_normalized - ALPHA_FINE) / ALPHA_FINE
            })
        
        # Shell-to-shell entropy ratios
        if verbose:
            print("[9.2] Computing entropy ratios...")
        
        for n in range(1, min(5, self.lattice.n_max + 1)):
            N_n = 2 * n**2
            N_next = 2 * (n+1)**2
            
            S_n = np.log(N_n)
            S_next = np.log(N_next)
            
            ratio = S_n / S_next if S_next > 0 else 0
            
            results['candidates'].append({
                'description': f'S(n={n})/S(n={n+1})',
                'value': ratio,
                'deviation_from_alpha': abs(ratio - ALPHA_FINE),
                'relative_error': abs(ratio - ALPHA_FINE) / ALPHA_FINE
            })
        
        # Mutual information placeholder
        if verbose:
            print("[9.3] Mutual information analysis...")
            print("  [Placeholder] Requires density matrix calculation")
        
        return results
    
    def track10_asymptotic(self, verbose=False):
        """
        Track 10: Asymptotic Expansion Analysis
        
        Analyze large-ℓ expansions.
        """
        if verbose:
            print("\n[10.1] Asymptotic expansion analysis...")
        
        results = {
            'track': 10,
            'name': 'Asymptotic Expansion',
            'candidates': []
        }
        
        # From Phase 6: convergence rate α = 0.19
        alpha_convergence = 0.19
        
        # Ratio to fine structure constant
        ratio = alpha_convergence / ALPHA_FINE
        results['candidates'].append({
            'description': 'α_convergence / α_fine',
            'value': ratio,
            'deviation_from_alpha': abs(ratio - ALPHA_FINE),
            'relative_error': abs(ratio - ALPHA_FINE) / ALPHA_FINE
        })
        
        # Inverse ratio
        inv_ratio = ALPHA_FINE / alpha_convergence
        results['candidates'].append({
            'description': 'α_fine / α_convergence',
            'value': inv_ratio,
            'deviation_from_alpha': abs(inv_ratio - ALPHA_FINE),
            'relative_error': abs(inv_ratio - ALPHA_FINE) / ALPHA_FINE
        })
        
        # Large-ℓ corrections
        if verbose:
            print("[10.2] Computing 1/ℓ expansion coefficients...")
        
        for ℓ in range(3, min(10, self.lattice.ℓ_max + 1)):
            # 1/ℓ series coefficients
            corrections = {
                '1/ℓ': 1/ℓ,
                '1/ℓ²': 1/ℓ**2,
                '1/√ℓ': 1/np.sqrt(ℓ),
                'ℓ/(ℓ²+1)': ℓ/(ℓ**2 + 1),
            }
            
            for desc, value in corrections.items():
                results['candidates'].append({
                    'description': f'{desc} for ℓ={ℓ}',
                    'value': value,
                    'deviation_from_alpha': abs(value - ALPHA_FINE),
                    'relative_error': abs(value - ALPHA_FINE) / ALPHA_FINE
                })
        
        return results
    
    def synthesize_results(self):
        """
        Synthesize findings from all tracks.
        
        Returns
        -------
        dict
            Best candidates and statistical analysis
        """
        all_candidates = []
        
        for track_name, track_results in self.results.items():
            if 'candidates' in track_results:
                for candidate in track_results['candidates']:
                    candidate['track'] = track_name
                    all_candidates.append(candidate)
        
        if not all_candidates:
            return {'best_candidates': [], 'statistics': {}}
        
        # Sort by relative error
        sorted_candidates = sorted(all_candidates, key=lambda x: x['relative_error'])
        
        # Top 20 candidates
        top_candidates = sorted_candidates[:20]
        
        # Statistics
        errors = [c['relative_error'] for c in all_candidates if np.isfinite(c['relative_error'])]
        
        statistics = {
            'total_candidates': len(all_candidates),
            'mean_relative_error': np.mean(errors) if errors else np.inf,
            'median_relative_error': np.median(errors) if errors else np.inf,
            'min_relative_error': np.min(errors) if errors else np.inf,
            'within_1_percent': sum(1 for e in errors if e < 0.01),
            'within_5_percent': sum(1 for e in errors if e < 0.05),
            'within_10_percent': sum(1 for e in errors if e < 0.10),
            'within_50_percent': sum(1 for e in errors if e < 0.50),
        }
        
        return {
            'best_candidates': top_candidates,
            'statistics': statistics
        }
    
    def _print_track_summary(self, name, result):
        """Print summary of track results."""
        if 'status' in result:
            print(f"\nStatus: {result['status']}")
            return
        
        if 'candidates' not in result or not result['candidates']:
            print("\nNo candidates found")
            return
        
        # Find best candidate
        best = min(result['candidates'], key=lambda x: x['relative_error'])
        
        print(f"\nBest candidate: {best['description']}")
        print(f"  Value: {best['value']:.6f}")
        print(f"  α = {ALPHA_FINE:.6f}")
        print(f"  Relative error: {best['relative_error']*100:.2f}%")
        
        # Count promising ones
        promising = sum(1 for c in result['candidates'] if c['relative_error'] < 0.50)
        print(f"\nPromising candidates (< 50% error): {promising}/{len(result['candidates'])}")
    
    def _print_synthesis(self, synthesis):
        """Print synthesis results."""
        stats = synthesis['statistics']
        
        print(f"\nTotal candidates evaluated: {stats['total_candidates']}")
        print(f"Within 1% of α: {stats['within_1_percent']}")
        print(f"Within 5% of α: {stats['within_5_percent']}")
        print(f"Within 10% of α: {stats['within_10_percent']}")
        print(f"Within 50% of α: {stats['within_50_percent']}")
        
        print(f"\n{'='*70}")
        print("TOP 10 CANDIDATES")
        print('='*70)
        
        for i, candidate in enumerate(synthesis['best_candidates'][:10], 1):
            print(f"\n{i}. {candidate['description']}")
            print(f"   Track: {candidate['track']}")
            print(f"   Value: {candidate['value']:.8f}")
            print(f"   α = {ALPHA_FINE:.8f}")
            print(f"   Relative error: {candidate['relative_error']*100:.4f}%")
    
    def plot_results(self, save_path=None):
        """
        Create visualization of all results.
        
        Parameters
        ----------
        save_path : str, optional
            Path to save the figure
        """
        if 'Synthesis' not in self.results:
            print("Run explore_all() first")
            return
        
        synthesis = self.results['Synthesis']
        candidates = synthesis['best_candidates'][:30]  # Top 30
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Best candidates
        ax = axes[0, 0]
        values = [c['value'] for c in candidates]
        errors = [c['relative_error'] for c in candidates]
        
        ax.scatter(range(len(values)), values, c=errors, cmap='RdYlGn_r', 
                   s=100, alpha=0.7, edgecolors='black')
        ax.axhline(ALPHA_FINE, color='red', linestyle='--', linewidth=2, label='α = 1/137.036')
        ax.set_xlabel('Candidate rank', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title('Top 30 Candidates', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Error distribution
        ax = axes[0, 1]
        all_errors = [c['relative_error'] for c in synthesis['best_candidates'] if c['relative_error'] < 5.0]
        ax.hist(all_errors, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(0.01, color='green', linestyle='--', label='1% error')
        ax.axvline(0.05, color='orange', linestyle='--', label='5% error')
        ax.axvline(0.10, color='red', linestyle='--', label='10% error')
        ax.set_xlabel('Relative error', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Error Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.set_yscale('log')
        
        # Plot 3: By track
        ax = axes[1, 0]
        track_names = []
        track_best_errors = []
        
        for track_name, track_result in self.results.items():
            if track_name == 'Synthesis':
                continue
            if 'candidates' in track_result and track_result['candidates']:
                best_error = min(c['relative_error'] for c in track_result['candidates'])
                track_names.append(track_name.split(':')[0].replace('Track ', 'T'))
                track_best_errors.append(best_error * 100)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(track_names)))
        ax.barh(track_names, track_best_errors, color=colors, edgecolor='black')
        ax.axvline(1, color='green', linestyle='--', linewidth=2, label='1% error')
        ax.axvline(5, color='orange', linestyle='--', linewidth=2, label='5% error')
        ax.axvline(10, color='red', linestyle='--', linewidth=2, label='10% error')
        ax.set_xlabel('Best relative error (%)', fontsize=12)
        ax.set_title('Best Result by Track', fontsize=14, fontweight='bold')
        ax.legend()
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Plot 4: Summary statistics
        ax = axes[1, 1]
        ax.axis('off')
        
        stats = synthesis['statistics']
        summary_text = f"""
        FINE STRUCTURE CONSTANT SEARCH
        α = 1/137.035999084
        
        Total candidates: {stats['total_candidates']}
        
        Accuracy levels:
        • Within 1%:  {stats['within_1_percent']} candidates
        • Within 5%:  {stats['within_5_percent']} candidates
        • Within 10%: {stats['within_10_percent']} candidates
        • Within 50%: {stats['within_50_percent']} candidates
        
        Best candidate:
        {candidates[0]['description'][:50]}...
        Value: {candidates[0]['value']:.8f}
        Error: {candidates[0]['relative_error']*100:.4f}%
        """
        
        ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        plt.show()
        
        return fig
