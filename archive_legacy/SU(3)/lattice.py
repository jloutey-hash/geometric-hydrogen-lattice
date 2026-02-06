"""
SU(3) Lattice Construction using Gelfand-Tsetlin Patterns
Generates the complete basis for SU(3) irreducible representations.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple


class SU3Lattice:
    """
    Generates and manages the weight states for SU(3) irreducible representations
    using Gelfand-Tsetlin (GT) patterns.
    
    GT patterns provide a unique labeling for every state, resolving multiplicity issues.
    """
    
    def __init__(self, max_p: int, max_q: int):
        """
        Initialize the SU(3) lattice using Gelfand-Tsetlin patterns.
        
        Parameters:
        -----------
        max_p : int
            Maximum value of the first Dynkin label
        max_q : int
            Maximum value of the second Dynkin label
        """
        self.max_p = max_p
        self.max_q = max_q
        self.states = []
        self.state_lookup = {}  # Maps GT pattern tuple to index
        
        # Generate all states
        self._generate_states()
        
    def _generate_states(self):
        """Generate all states using Gelfand-Tsetlin patterns."""
        index = 0
        
        for p in range(self.max_p + 1):
            for q in range(self.max_q + 1):
                # Generate all GT patterns for this (p, q) representation
                gt_patterns = self._generate_gt_patterns(p, q)
                
                for gt in gt_patterns:
                    # Compute (I3, Y) coordinates from GT pattern
                    i3, y = self._gt_to_weight(gt)
                    
                    state = {
                        'p': p,
                        'q': q,
                        'gt': gt,  # Store the GT pattern
                        'm13': gt[0], 'm23': gt[1], 'm33': gt[2],
                        'm12': gt[3], 'm22': gt[4],
                        'm11': gt[5],
                        'i3': i3,
                        'y': y,
                        'index': index
                    }
                    self.states.append(state)
                    self.state_lookup[gt] = index
                    index += 1
    
    def _generate_gt_patterns(self, p: int, q: int) -> List[Tuple[int, ...]]:
        """
        Generate all valid Gelfand-Tsetlin patterns for the (p, q) representation.
        
        A GT pattern for SU(3) is a triangular array:
            m13  m23  m33
              m12  m22
                m11
        
        With constraints:
        - Top row: (m13, m23, m33) = (p+q, q, 0) for representation (p, q)
        - Betweenness: m13 >= m12 >= m23 >= m22 >= m33
        - Betweenness: m12 >= m11 >= m22
        
        Parameters:
        -----------
        p, q : int
            Dynkin labels
            
        Returns:
        --------
        List of tuples (m13, m23, m33, m12, m22, m11)
        """
        m13 = p + q
        m23 = q
        m33 = 0
        
        patterns = []
        
        # Iterate over all valid middle row values
        for m12 in range(m23, m13 + 1):
            for m22 in range(m33, m23 + 1):
                # Iterate over all valid bottom row values
                for m11 in range(m22, m12 + 1):
                    pattern = (m13, m23, m33, m12, m22, m11)
                    patterns.append(pattern)
        
        return patterns
    
    def _gt_to_weight(self, gt: Tuple[int, ...]) -> Tuple[float, float]:
        """
        Convert a Gelfand-Tsetlin pattern to (I3, Y) weight coordinates.
        
        Formulas:
        I3 = m11 - (m12 + m22) / 2
        Y = (m12 + m22) - 2(m13 + m23 + m33) / 3
        
        Parameters:
        -----------
        gt : tuple
            GT pattern (m13, m23, m33, m12, m22, m11)
            
        Returns:
        --------
        (i3, y) : tuple of floats
        """
        m13, m23, m33, m12, m22, m11 = gt
        
        i3 = m11 - (m12 + m22) / 2.0
        y = (m12 + m22) - 2 * (m13 + m23 + m33) / 3.0
        
        return i3, y
    
    def get_index(self, gt: Tuple[int, ...]) -> Optional[int]:
        """
        Get the unique index for a given GT pattern.
        
        Parameters:
        -----------
        gt : tuple
            GT pattern (m13, m23, m33, m12, m22, m11)
            
        Returns:
        --------
        int or None
            The state index, or None if state doesn't exist
        """
        return self.state_lookup.get(gt, None)
    
    def get_index_by_weight(self, p: int, q: int, i3: float, y: float) -> List[int]:
        """
        Get all state indices that have the given (p, q, I3, Y) values.
        Useful for checking multiplicity.
        
        Parameters:
        -----------
        p, q : int
            Dynkin labels
        i3, y : float
            Weight coordinates
            
        Returns:
        --------
        List of indices
        """
        indices = []
        for state in self.states:
            if (state['p'] == p and state['q'] == q and
                abs(state['i3'] - i3) < 1e-6 and abs(state['y'] - y) < 1e-6):
                indices.append(state['index'])
        return indices
    
    def get_state(self, index: int) -> Dict:
        """Get state dictionary by index."""
        if 0 <= index < len(self.states):
            return self.states[index]
        return None
    
    def get_dimension(self) -> int:
        """Get total number of states in the lattice."""
        return len(self.states)
    
    def get_coordinates(self) -> np.ndarray:
        """
        Get all (I3, Y) coordinates as a 2D array.
        
        Returns:
        --------
        np.ndarray of shape (N, 2)
            Array where each row is [i3, y]
        """
        coords = np.array([[s['i3'], s['y']] for s in self.states])
        return coords
    
    def print_summary(self):
        """Print a summary of the lattice."""
        print(f"SU(3) Lattice Summary (Gelfand-Tsetlin Basis)")
        print(f"=" * 50)
        print(f"Max (p, q): ({self.max_p}, {self.max_q})")
        print(f"Total states: {len(self.states)}")
        print(f"\nRepresentations included:")
        
        reps = {}
        for state in self.states:
            key = (state['p'], state['q'])
            reps[key] = reps.get(key, 0) + 1
        
        for (p, q), count in sorted(reps.items()):
            dim_theory = self._representation_dimension(p, q)
            match = "✓" if count == dim_theory else "✗"
            print(f"  ({p}, {q}): {count} states (theory: {dim_theory}) {match}")
    
    @staticmethod
    def _representation_dimension(p: int, q: int) -> int:
        """Calculate the dimension of the (p, q) representation."""
        return (p + 1) * (q + 1) * (p + q + 2) // 2


if __name__ == "__main__":
    # Test the lattice construction
    lattice = SU3Lattice(max_p=2, max_q=2)
    lattice.print_summary()
    
    print("\nSample states (first 10):")
    for i, state in enumerate(lattice.states[:10]):
        gt = state['gt']
        print(f"  {i}: (p={state['p']}, q={state['q']}) GT={gt} -> "
              f"I3={state['i3']:.2f}, Y={state['y']:.2f}")
    
    # Check a specific representation
    print("\n(2,1) representation states:")
    states_21 = [s for s in lattice.states if s['p'] == 2 and s['q'] == 1]
    print(f"Found {len(states_21)} states (expected 15)")
    for s in states_21[:5]:
        print(f"  GT={s['gt']}, I3={s['i3']:.2f}, Y={s['y']:.2f}")
