"""
Quantum-Geometric Lattice Construction Package

This package implements a discrete 2D polar lattice that reproduces
the degeneracy structure of hydrogen atom quantum states.
"""

from .lattice import PolarLattice
from .operators import LatticeOperators
from .angular_momentum import AngularMomentumOperators
from .quantum_comparison import QuantumComparison
from .spin import SpinOperators, ShellFilling
from .convergence import ConvergenceAnalysis, RydbergAnalysis
from .visualization import LatticeVisualizer, ComparisonDashboard, DocumentationGenerator

__version__ = "1.0.0"
__all__ = ["PolarLattice", "LatticeOperators", "AngularMomentumOperators", 
           "QuantumComparison", "SpinOperators", "ShellFilling",
           "ConvergenceAnalysis", "RydbergAnalysis",
           "LatticeVisualizer", "ComparisonDashboard", "DocumentationGenerator"]
