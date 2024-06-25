# std lib dependencies
from typing import List, Tuple

# third party dependencies
import numpy as np

# project dependencies
from common_dtypes import GomoryCut
from policies import BaseCutSelectPolicy


class BaseCutSelectAgent:
    def __init__(self, policy: BaseCutSelectPolicy, mode: str = "cutselect"):
        """Policy.act recieves a list of cuts and LP data and returns the list of (policy.ncuts) selected cuts."""
        self.policy = policy
        self.mode = mode
        assert self.mode in [
            "cutselect",
            "reselect",
        ], f"Mode should be in ['cutselect', 'reselect'] but is {mode}"

    def cutselect(
        self,
        reserved_m: int,
        n_cuts_added_previous_iterations: int,
        cuts: List[GomoryCut],
        A: np.ndarray,
        b: np.array,
        c: np.array,
        x: np.array,
        SimplexTableau: np.ndarray,
        SimplexSolution: np.ndarray,
    ) -> Tuple[List[GomoryCut], List[GomoryCut]]:
        """Perfoms a cutselect step depending on the specified mode.
        Supported modes:
        - 'cutselect': select the best ncuts from the list of cuts
        - 'reselect': re-select the best ncuts from the list of new cuts and the previous cuts (non reserved in [A,b])
        """
        assert self.mode in [
            "cutselect",
            "reselect",
        ], f"Mode should be in ['cutselect', 'reselect'] but is {mode}"
        if self.mode == "cutselect":
            return self._cutselect(
                reserved_m=reserved_m,
                cuts=cuts,
                A=A,
                b=b,
                c=c,
                x=x,
                SimplexTableau=SimplexTableau,
                SimplexSolution=SimplexSolution,
            )
        elif self.mode == "reselect":
            return self._reselect(
                reserved_m=reserved_m,
                n_cuts_added_previous_iterations=n_cuts_added_previous_iterations,
                cuts=cuts,
                A=A,
                b=b,
                c=c,
                x=x,
                SimplexTableau=SimplexTableau,
                SimplexSolution=SimplexSolution,
            )

    def _cutselect(
        self,
        reserved_m: int,
        cuts: List[GomoryCut],
        A: np.ndarray,
        b: np.array,
        c: np.array,
        x: np.array,
        SimplexTableau: np.ndarray,
        SimplexSolution: np.ndarray,
    ) -> Tuple[List[GomoryCut], List[GomoryCut]]:
        """Select the best ncuts from the list of cuts. This acting method does not remove any cuts from the ones added in previous iterations."""
        return (
            self.policy.act(
                cuts,
                A=A,
                b=b,
                c=c,
                x=x,
                SimplexTableau=SimplexTableau,
                SimplexSolution=SimplexSolution,
            ),
            [],
        )

    def _reselect(
        self,
        reserved_m: int,
        n_cuts_added_previous_iterations: int,
        cuts: List[GomoryCut],
        A: np.ndarray,
        b: np.array,
        c: np.array,
        x: np.array,
        SimplexTableau: np.ndarray,
        SimplexSolution: np.ndarray,
    ) -> Tuple[List[GomoryCut], List[GomoryCut]]:
        """Re-select the best ncuts from the list of new cuts and the previous cuts (non reserved in [A,b])"""
        assert cuts == [], f"cuts should be empty but is {cuts}"
        previous_cuts = self.get_previous_added_cuts(reserved_m=reserved_m, A=A, b=b)
        # NOTE that if we allow maxcuts as the full candidates this could make cuts smaller than the cuts to select
        all_cuts = self.remove_duplicate_cuts(previous_cuts)
        cut_increase_per_iteration = self.policy.ncuts
        self.policy.ncuts = (
            n_cuts_added_previous_iterations + cut_increase_per_iteration
        )
        selected_cuts = self.policy.act(
            cuts=all_cuts,
            A=A,
            b=b,
            c=c,
            x=x,
            SimplexTableau=SimplexTableau,
            SimplexSolution=SimplexSolution,
        )
        self.policy.ncuts = cut_increase_per_iteration
        cuts_to_add, cuts_to_remove = (
            self.separate_selected_cuts_in_cuts_to_add_cuts_to_remove(
                selected_cuts=selected_cuts, previous_cuts=all_cuts, current_cuts=cuts
            )
        )

        return cuts_to_add, cuts_to_remove

    def separate_selected_cuts_in_cuts_to_add_cuts_to_remove(
        self,
        selected_cuts: List[GomoryCut],
        previous_cuts: List[GomoryCut],
        current_cuts: List[GomoryCut],
    ) -> Tuple[List[GomoryCut], List[GomoryCut]]:
        """Given the list of cuts and the list of selected cuts, returns the list of cuts to add and the list of cuts to remove. This separation is done only with the coefficients of the cuts."""
        cuts_to_add = []
        cuts_to_remove = previous_cuts.copy()
        for cut in selected_cuts:
            if self.is_cut_in_cutlist(cut, current_cuts):
                cuts_to_add.append(cut)
            elif self.is_cut_in_cutlist(cut, previous_cuts):
                cuts_to_remove = self.remove_cut_from_cutlist(cut, cuts_to_remove)
            else:
                raise ValueError(
                    f"Cut {cut} not found in current_cuts nor previous_cuts"
                )
        return cuts_to_add, cuts_to_remove

    @staticmethod
    def get_previous_added_cuts(
        reserved_m: int, A: np.ndarray, b: np.array
    ) -> List[GomoryCut]:
        """Returns the list of cuts added in the previous iterations (non reserved in [A,b])."""
        A_non_reserved = A[reserved_m:, :]
        b_non_reserved = b[reserved_m:]
        cuts = []
        for i in range(len(A_non_reserved)):
            cuts.append(
                GomoryCut(
                    coefficients=A_non_reserved[i],
                    rhs=b_non_reserved[i],
                    tableau_idx=None,
                    nonbasic_var_idx=None,
                )
            )
        return cuts

    @staticmethod
    def is_cut_in_cutlist(cut: GomoryCut, cutlist: List[GomoryCut]) -> bool:
        """Returns True if the cut is in the cutlist based on a match in the coefficients and rhs."""
        cut_coefficients_and_rhs = [*cut.coefficients, cut.rhs]
        for cut_in_cutlist in cutlist:
            cut_in_cutlist_coefficients_and_rhs = [
                *cut_in_cutlist.coefficients,
                cut_in_cutlist.rhs,
            ]
            if cut_coefficients_and_rhs == cut_in_cutlist_coefficients_and_rhs:
                return True
        return False

    @staticmethod
    def remove_cut_from_cutlist(
        cut: GomoryCut, cutlist: List[GomoryCut]
    ) -> List[GomoryCut]:
        """Returns the cutlist without the cut."""
        cut_coefficients_and_rhs = [*cut.coefficients, cut.rhs]
        orig_lenght = len(cutlist)
        filtered_cutlist = [
            c for c in cutlist if [*c.coefficients, c.rhs] != cut_coefficients_and_rhs
        ]
        try:
            assert (
                len(filtered_cutlist) < orig_lenght
            ), f"Cut {cut} not found in cutlist"  # note that it is < as there can be duplicate cuts in cutlist
        except AssertionError as e:
            print(e)
            import pdb

            pdb.set_trace()
        return filtered_cutlist

    @staticmethod
    def remove_duplicate_cuts(cuts: List[GomoryCut]) -> List[GomoryCut]:
        """Returns the list of cuts without duplicates. Two cuts are considered equal if they have the same coefficients and rhs."""
        cuts_coefficients_and_rhs = [[*c.coefficients, c.rhs] for c in cuts]
        cuts_coefficients_and_rhs_unique = list(
            set(tuple(c) for c in cuts_coefficients_and_rhs)
        )
        cuts_unique = []
        for cut_coefficients_and_rhs in cuts_coefficients_and_rhs_unique:
            # find the tableau_idx and nonbasic_var_idx of the cut
            for cut in cuts:
                if tuple([*cut.coefficients, cut.rhs]) == cut_coefficients_and_rhs:
                    cut_coefficients_and_rhs = list(cut_coefficients_and_rhs)
                    cut = GomoryCut(
                        coefficients=cut_coefficients_and_rhs[:-1],
                        rhs=cut_coefficients_and_rhs[-1],
                        tableau_idx=cut.tableau_idx,
                        nonbasic_var_idx=cut.nonbasic_var_idx,
                    )
                    cuts_unique.append(cut)
                    break
            else:
                raise ValueError(f"Cut {cut_coefficients_and_rhs} not found in cuts")
        return cuts_unique
