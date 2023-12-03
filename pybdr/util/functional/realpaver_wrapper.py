"""
REF:

http://pagesperso.lina.univ-nantes.fr/~granvilliers-l/realpaver/src/realpaver-0.4.pdf

"""

import re
import subprocess
import tempfile

import numpy as np

from pybdr.geometry import Interval
from pybdr.util.functional.auxiliary import get_system

_constraint_pat = r"([a-zA-Z\d_]+)\s+in\s+(\[|\])(.*),(.*)(\]|\[)"
_box_name_pat = r"(INNER|OUTER|INITIAL)\sBOX(\s\d+)*"


class Constant:
    def __init__(self, var_name, value):
        self._var_name = var_name
        self._value = value

    def to_input(self):
        return str(self._var_name) + " = " + str(self._value)


class Variable:
    def __init__(
            self, var_name, lower_bound, upper_bound, lower_bracket, upper_bracket
    ):
        assert lower_bound <= upper_bound
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._var_name = var_name
        self._lower_bracket = "["
        self._upper_bracket = "]"

        if np.isinf(lower_bound):
            self._lower_bound = "-oo"
            self._lower_bracket = "]"
        if np.isinf(upper_bound):
            self._upper_bound = "+oo"
            self._upper_bracket = "["

    def to_input(self):
        ret = self._var_name + " in "
        ret += self._lower_bracket + str(self._lower_bound) + ", "
        ret += str(self._upper_bound) + self._upper_bracket
        return ret


class RealPaver:
    def __init__(self):
        self.__constants = []
        self.__variables = []
        self.__constraints = []
        self.__input = None
        self._output_digits = 16
        self._output_mode = "union"  # 'union' or 'hull'
        self._output_style = "bound"  # 'bound' or 'midpoint'
        self._search_choice = "lf"  # 'rr' or 'lf' or mn
        self._search_parts = 2  # 2 or 3
        self._search_precision = 1e-8  # default 10^-8
        self._search_mode = "paving"  # 'paving' or 'points'
        self._search_number = 1024  # 'natural number n' or +oo

    def _build(self):
        self.__input = ""
        # flags for the search
        self.__input += "Branch\n"
        self.__input += "choice = " + self._search_choice + ",\n"
        self.__input += "parts = " + str(self._search_parts) + ",\n"
        self.__input += (
                "precision = " + "{:.20e}".format(self._search_precision) + ",\n"
        )
        self.__input += "mode = " + self._search_mode + ",\n"
        self.__input += "number = "
        self.__input += (
            str(self._search_number)
            if isinstance(self._search_number, int)
            else self._search_number
        )
        self.__input += " ;"

        # flags for the output
        self.__input += "\n\n"
        self.__input += "Output\n"
        self.__input += "digits = " + str(self._output_digits) + ",\n"
        self.__input += "mode = " + self._output_mode + ",\n"
        self.__input += "style = " + self._output_style + " ;"

        # define constants
        if len(self.__constants) > 0:
            self.__input += "\n\n"
            self.__input += "Constants\n"
            input_constants = ", \n".join(
                [" " + c.to_input() for c in self.__constants]
            )
            self.__input += input_constants + " ;"

        # define variables
        self.__input += "\n\n"
        self.__input += "Variables\n"
        input_variables = ", \n".join([" " + v.to_input() for v in self.__variables])
        self.__input += input_variables + " ;"

        # define constraints
        self.__input += "\n\n"
        self.__input += "Constraints\n"
        input_constraints = ", \n".join([" " + c for c in self.__constraints])
        self.__input += input_constraints + " ;"

    def _get_bin_path(self):
        import os
        from pathlib import Path

        this_path = os.path.dirname(__file__)
        this_sys = get_system()
        bin_name = "realpaver_"
        if this_sys == "linux":
            bin_name += "linux"
        elif this_sys == "macos":
            bin_name += "mac"
        elif this_sys == "windows":
            bin_name += "windows"
        else:
            raise Exception("invalid system for realpaver!!!")
        return Path(this_path, "bin", bin_name)

    def _solve(self):
        assert self.__input is not None

        with tempfile.NamedTemporaryFile("wt") as file:
            file.write(self.__input)
            file.flush()

            bin_path = self._get_bin_path()
            cmd = [bin_path, file.name]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return proc.stdout

    def _parse_single_box(self, s: str):
        matched = re.search(_box_name_pat, s)
        if matched:
            inner_outer = matched.group(1)
            box_id = matched.group(2)
            return inner_outer, int(box_id)
        else:
            return None, None

    def _parse_solutions(self, output):
        boxes = []
        lines = output.splitlines()
        for idx in range(len(lines)):
            if "BOX" in lines[idx]:
                if "INITIAL" in lines[idx]:
                    while "BOX" not in lines[idx]:
                        idx += 1
                    continue
                else:
                    box_type, box_id = self._parse_single_box(lines[idx])
                    idx += 1
                    data = []
                    while "BOX" not in lines[idx]:
                        matched = re.search(_constraint_pat, lines[idx])
                        if matched is not None:
                            variable = matched.group(1)
                            lower = float(matched.group(3))
                            upper = float(matched.group(4))
                            data.append(lower)
                            data.append(upper)
                        idx += 1
                        if idx >= len(lines):
                            break
                    data = np.array(data).reshape((-1, 2))
                    boxes.append([box_type, box_id, Interval(data[:, 0], data[:, 1])])
        return boxes

    def solve(self):
        self._build()
        output = self._solve()
        return self._parse_solutions(output)

    def add_constant(self, name, value: float):
        self.__constants.append(Constant(name, value))

    def add_variable(
            self, name: str, lower_bound, upper_bound, lower_bracket, upper_bracket
    ):
        self.__variables.append(
            Variable(name, lower_bound, upper_bound, lower_bracket, upper_bracket)
        )

    def add_constraint(self, constraint: str):
        self.__constraints.append(constraint)

    def set_output(self, digits=16, mode="union", style="bound"):
        self._output_digits = digits
        self._output_mode = mode
        self._output_style = style

    def set_branch(
            self, choice="lf", parts=2, precision=1e-8, mode="paving", number=1024
    ):
        self._search_choice = choice
        self._search_parts = parts
        self._search_precision = precision
        self._search_mode = mode
        self._search_number = number
