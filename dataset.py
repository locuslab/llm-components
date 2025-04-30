import random
import numpy as np
from tqdm import tqdm


class TwoDigitArithmeticDataset:
    def __init__(self, arithmetic_type, include_reversed=False):
        """
        Initializes the dataset.

        Args:
            include_reversed (bool, optional): If True, ensures that for each pair
                                               (a, b) where a != b, both the prompts
                                               "a × b=" and "b × a=" are included
                                               in the dataset (each exactly once).
                                               Defaults to False, which includes only
                                               "a × b=" for each pair encountered.
        """
        self.include_reversed = include_reversed
        print(f"Generating data for range [0, 999] with include_reversed={include_reversed}...")
        print("(Using set to ensure uniqueness and avoid double counting)")

        temp_data = []
        pairs_set = set() # Use a set to track added pairs for uniqueness

        for i in range(100):
            for j in range(100):
                pair = (i, j)
                reversed_pair = (j, i)

                # Add the primary pair (i, j) if it hasn't been added
                if pair not in pairs_set:
                    temp_data.append(pair)
                    pairs_set.add(pair)

                # If reversed is requested and i!=j, ensure the reversed pair is also added
                if include_reversed and i != j:
                    if reversed_pair not in pairs_set:
                        temp_data.append(reversed_pair)
                        pairs_set.add(reversed_pair)

        self.data = temp_data # Assign the unique pairs list
        print(f"Data generation complete. Total unique pairs: {len(self.data)}")

        # Generate prompts and corresponding answers from the final unique data
        print("Generating prompts and answers...")
        if arithmetic_type == "multiplication":
            self.prompts = [f"{a} x {b} =" for a, b in self.data]
            self.answers = [str(a * b) for a, b in self.data]
        elif arithmetic_type == "addition":
            self.prompts = [f"{a} + {b} =" for a, b in self.data]
            self.answers = [str(a + b) for a, b in self.data]
        elif arithmetic_type == "subtraction":
            self.prompts = [f"{a} - {b} =" for a, b in self.data]
            self.answers = [str(a - b) for a, b in self.data]
        else:
            raise ValueError(f"Invalid arithmetic type: {arithmetic_type}")

        print("Prompts and answers generated.")

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        try:
            return {
                "input": self.prompts[idx],
                "target": self.answers[idx]
            }
        except IndexError as e:
             raise IndexError(f"Dataset index {idx} out of range for size {len(self.prompts)}") from e


class SingleDigitMultiplicationDataset:
    def __init__(self, include_reversed=False):
        self.data = []
        for i in range(10):
            for j in range(10):
                self.data.append((i, j))
                if include_reversed and i != j:
                    self.data.append((j, i))
        self.prompts = [f"{a} × {b}=" for a, b in self.data]
        self.answers = [str(a * b) for a, b in self.data]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {
            "input": self.prompts[idx],
            "target": self.answers[idx]
        }