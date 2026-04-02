"""
Solution Script: Rule-based + ML-based hybrid approach
For Nemotron Reasoning Challenge
Uses pattern matching for bit manipulation and text encryption
"""

import pandas as pd
import re
import json

# Load data
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

print(f"Training samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")


# Analyze puzzle types and learn transformation rules
def analyze_bit_puzzle(prompt):
    """Learn the bit manipulation rule from examples."""
    # Extract input-output pairs
    pattern = r"([01]{8})\s*->\s*([01]{8})"
    examples = re.findall(pattern, prompt)

    if not examples:
        return None

    # Try various operations
    def apply_op(inp, op_name):
        inp_int = int(inp, 2)
        if op_name == "reverse":
            return format(int(bin(inp_int)[2:].zfill(8)[::-1], 2), "08b")
        elif op_name == "complement":
            return format((~inp_int) & 0xFF, "08b")
        elif op_name == "swap_nibbles":
            return format(((inp_int & 0xF0) >> 4) | ((inp_int & 0x0F) << 4), "08b")
        elif op_name == "rotate_left_1":
            return format(((inp_int << 1) | (inp_int >> 7)) & 0xFF, "08b")
        elif op_name == "rotate_right_1":
            return format(((inp_int >> 1) | (inp_int << 7)) & 0xFF, "08b")
        return None

    # Check which operation works
    for op in [
        "reverse",
        "complement",
        "swap_nibbles",
        "rotate_left_1",
        "rotate_right_1",
    ]:
        matches = 0
        for inp, out in examples:
            if apply_op(inp, op) == out:
                matches += 1
        if matches == len(examples) and len(examples) > 0:
            return op

    # Try XOR with constant
    for const in range(0xFF):
        matches = 0
        for inp, out in examples:
            inp_int = int(inp, 2)
            if format(inp_int ^ const, "08b") == out:
                matches += 1
        if matches == len(examples):
            return f"XOR_{hex(const)}"

    return "unknown"


def solve_bit_puzzle(prompt):
    """Solve a bit manipulation puzzle."""
    # Extract examples
    pattern = r"([01]{8})\s*->\s*([01]{8})"
    examples = re.findall(pattern, prompt)

    # Find target
    target_match = re.search(r"Now, determine the output for:\s*([01]{8})", prompt)
    if not target_match:
        return None

    target = target_match.group(1)

    if not examples:
        return None

    # Find matching operation
    ops = {
        "reverse": lambda x: int(bin(x)[2:].zfill(8)[::-1], 2),
        "complement": lambda x: (~x) & 0xFF,
        "swap_nibbles": lambda x: ((x & 0xF0) >> 4) | ((x & 0x0F) << 4),
        "rotate_left_1": lambda x: ((x << 1) | (x >> 7)) & 0xFF,
        "rotate_right_1": lambda x: ((x >> 1) | (x << 7)) & 0xFF,
    }

    for name, op in ops.items():
        matches = 0
        for inp, out in examples:
            if format(op(int(inp, 2)), "08b") == out:
                matches += 1
        if matches == len(examples):
            result = op(int(target, 2))
            return format(result, "08b")

    # Try simple per-bit transformations
    # For each position, determine transformation from examples
    def solve_from_examples(target, examples):
        # Build truth table for each bit position
        results = []
        for pos in range(8):
            in_bits = [e[0][pos] for e in examples]
            out_bits = [e[1][pos] for e in examples]

            # Check if there's a consistent transformation
            # Could be: identity, NOT, or same as another bit
            transformed = []
            for bit in target[pos]:
                # Try identity
                if (bit, bit) in zip(in_bits, out_bits):
                    transformed.append(bit)
                    continue
                # Try NOT
                flipped = "1" if bit == "0" else "0"
                if (bit, flipped) in zip(in_bits, out_bits):
                    transformed.append(flipped)
                    continue

            results.append(transformed[pos] if pos < len(transformed) else target[pos])

        return "".join(results)

    return solve_from_examples(target, examples)


def analyze_text_encryption(prompt):
    """Learn text encryption mapping from examples."""
    lines = prompt.split("\n")
    examples = []

    for line in lines:
        if "->" in line and "Now" not in line:
            parts = line.split("->")
            if len(parts) == 2:
                encrypted = parts[0].strip()
                decrypted = parts[1].strip()
                examples.append((encrypted, decrypted))

    return examples


def solve_text_encryption(prompt):
    """Solve text encryption puzzle."""
    examples = analyze_text_encryption(prompt)

    if not examples:
        return None

    # Extract target
    target_match = re.search(r"Now, decrypt the following text:\s*(.+)", prompt)
    if not target_match:
        return None

    target = target_match.group(1).strip()

    # Build character mapping
    # Assuming consistent letter substitution
    encrypted_chars = set()
    decrypted_chars = set()

    for enc, dec in examples:
        encrypted_chars.update(enc.split())
        decrypted_chars.update(dec.split())

    # Create mapping based on word positions
    # This is a simplified approach
    enc_words = [e[0].split() for e in examples]
    dec_words = [e[1].split() for e in examples]

    # Map encrypted words to decrypted
    word_map = {}
    for ew, dw in zip(enc_words, dec_words):
        for e, d in zip(ew, dw):
            word_map[e] = d

    # Apply to target
    target_words = target.split()
    result = " ".join(word_map.get(w, w) for w in target_words)

    return result


# Analyze training data to understand patterns
print("\n=== Analyzing Bit Manipulation Rules ===")
bit_puzzles = train_df[train_df["prompt"].str.contains("bit manipulation", case=False)]

for i in range(min(5, len(bit_puzzles))):
    rule = analyze_bit_puzzle(bit_puzzles.iloc[i]["prompt"])
    print(f"Puzzle {i + 1}: Rule = {rule}")

# Solve test puzzles
print("\n=== Solving Test Puzzles ===")
results = []

for i, row in test_df.iterrows():
    prompt = row["prompt"]

    if "bit manipulation" in prompt.lower():
        answer = solve_bit_puzzle(prompt)
    elif "encrypt" in prompt.lower() or "decrypt" in prompt.lower():
        answer = solve_text_encryption(prompt)
    else:
        answer = "unknown"

    results.append({"id": row["id"], "answer": answer})
    print(f"Test {i + 1}: {answer}")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("submission_baseline.csv", index=False)
print("\nSaved baseline predictions to submission_baseline.csv")
