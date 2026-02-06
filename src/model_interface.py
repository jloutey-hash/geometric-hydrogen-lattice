"""
Unified Geometric Impedance Framework

This module defines a common interface for computing geometric impedance across
different gauge groups: U(1), SU(2), and SU(3).

The central concept: Coupling constants as geometric information-conversion ratios.

    κ = C_matter / S_gauge  (Paper convention: κ ≈ 137 for hydrogen)
    
    Equivalently: Z = S_gauge / C_matter = 1/κ  (Code historical convention)

Where:
- C_matter: Matter capacity (symplectic volume, state count, phase space)
- S_gauge: Gauge action (holonomy, Wilson loops, plaquette sums)
- κ: Impedance (dimensionless ratio, information conversion efficiency)

NOTE: This file now uses κ notation to match the papers.
Previous versions used Z = 1/κ, which caused confusion.

This framework extends the hydrogen/photon impedance concept (α ≈ 1/κ where κ = S/P)
to SU(2) and SU(3) gauge theories, treating coupling constants as geometric invariants.

Author: Computational Physics Research
Date: February 2026
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np


@dataclass
class ImpedanceResult:
    """
    Standard return type for impedance calculations across all systems.
    
    Fields:
    -------
    system_type : str
        Type of system: 'U(1)', 'SU(2)', 'SU(3)'
    label : str
        Human-readable label (e.g., "Hydrogen n=5", "SU(3) (3,0)")
    C_matter : float
        Matter capacity (symplectic volume, state count, etc.)
    S_gauge : float
        Gauge action (holonomy, Wilson loop sum, etc.)
    kappa_impedance : float
        Impedance ratio κ = C_matter / S_gauge (paper convention, κ ≈ 137 for hydrogen)
    size_parameter : float
        Characteristic size (n for hydrogen, j for SU(2), sqrt(p²+pq+q²) for SU(3))
    metadata : dict
        System-specific additional data
    """
    system_type: str
    label: str
    C_matter: float
    S_gauge: float
    kappa_impedance: float
    size_parameter: float
    metadata: Dict[str, Any]
    
    def __repr__(self):
        return (f"ImpedanceResult({self.system_type}: {self.label}, "
                f"κ={self.kappa_impedance:.6f}, C={self.C_matter:.2f}, S={self.S_gauge:.2f})")


class GeometricImpedanceSystem(ABC):
    """
    Abstract base class for geometric impedance calculations.
    
    All gauge systems (U(1), SU(2), SU(3)) must implement:
    1. compute_matter_capacity() -> float
    2. compute_gauge_action() -> float
    3. get_label() -> str
    4. get_size_parameter() -> float
    
    The impedance ratio is automatically computed as Z = S_gauge / C_matter.
    """
    
    @abstractmethod
    def compute_matter_capacity(self) -> float:
        """
        Compute matter symplectic capacity C_matter.
        
        Physical interpretation:
        - U(1): Symplectic volume of electron phase space (sum of plaquette areas)
        - SU(2): State count or angular momentum capacity
        - SU(3): Symplectic capacity of color representation
        
        Returns:
        --------
        C_matter : float
            Matter capacity (positive)
        """
        pass
    
    @abstractmethod
    def compute_gauge_action(self) -> float:
        """
        Compute gauge action S_gauge.
        
        Physical interpretation:
        - U(1): Photon fiber action (helical winding)
        - SU(2): SU(2) holonomy or plaquette sum
        - SU(3): Color holonomy (Wilson loops, plaquette traces)
        
        Returns:
        --------
        S_gauge : float
            Gauge action (positive)
        """
        pass
    
    def compute_impedance(self) -> float:
        """
        Compute geometric impedance κ = C_matter / S_gauge.
        
        This is the central quantity: the ratio of matter capacity to gauge action.
        Physically, it represents information conversion efficiency between
        matter and gauge degrees of freedom.
        
        NOTATION (Paper convention):
        For U(1) hydrogen: κ ~ 137 ≈ 1/α (fine structure constant)
        For SU(2): κ ~ ? (to be explored)
        For SU(3): κ ~ ? (QCD-inspired, not claiming α_s derivation)
        
        Historical note: Previous code used Z = S/C = 1/κ, causing confusion.
        This version standardizes to paper's κ = C/S notation.
        
        Returns:
        --------
        κ : float
            Impedance (dimensionless)
        """
        C = self.compute_matter_capacity()
        S = self.compute_gauge_action()
        
        if C <= 0:
            raise ValueError(f"Matter capacity must be positive, got {C}")
        if S <= 0:
            raise ValueError(f"Gauge action must be positive, got {S}")
        
        return C / S  # Paper convention: κ = C/S ≈ 137
    
    @abstractmethod
    def get_label(self) -> str:
        """
        Get human-readable label for this system.
        
        Examples:
        - "Hydrogen n=5"
        - "SU(2) j=3/2"
        - "SU(3) (3,0)"
        """
        pass
    
    @abstractmethod
    def get_size_parameter(self) -> float:
        """
        Get characteristic size parameter for plotting.
        
        Examples:
        - U(1): principal quantum number n
        - SU(2): spin j
        - SU(3): representation size sqrt(p² + pq + q²)
        """
        pass
    
    def get_system_type(self) -> str:
        """
        Get system type string.
        
        Returns one of: 'U(1)', 'SU(2)', 'SU(3)'
        """
        # Default: extract from class name
        class_name = self.__class__.__name__
        if 'U1' in class_name or 'Hydrogen' in class_name:
            return 'U(1)'
        elif 'SU2' in class_name:
            return 'SU(2)'
        elif 'SU3' in class_name:
            return 'SU(3)'
        else:
            return 'Unknown'
    
    def compute(self) -> ImpedanceResult:
        """
        Compute full impedance result with metadata.
        
        Returns:
        --------
        ImpedanceResult
            Complete result object with κ = C_matter / S_gauge
        """
        C_matter = self.compute_matter_capacity()
        S_gauge = self.compute_gauge_action()
        kappa_impedance = self.compute_impedance()
        
        return ImpedanceResult(
            system_type=self.get_system_type(),
            label=self.get_label(),
            C_matter=C_matter,
            S_gauge=S_gauge,
            kappa_impedance=kappa_impedance,
            size_parameter=self.get_size_parameter(),
            metadata=self.get_metadata()
        )
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get system-specific metadata.
        
        Override in subclasses to add custom fields.
        """
        return {}
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.get_label()})"


# ============================================================================
# Validation helpers
# ============================================================================

def validate_positive(value: float, name: str) -> None:
    """Check that a value is positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def validate_quantum_number(n: int, min_val: int = 1, max_val: Optional[int] = None) -> None:
    """Check that a quantum number is valid."""
    if not isinstance(n, int):
        raise TypeError(f"Quantum number must be integer, got {type(n)}")
    if n < min_val:
        raise ValueError(f"Quantum number must be >= {min_val}, got {n}")
    if max_val is not None and n > max_val:
        raise ValueError(f"Quantum number must be <= {max_val}, got {n}")
