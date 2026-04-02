Data is located in `./data`

This dataset comprises a collection of logical reasoning puzzles requiring the identification and application of underlying transformation rules. The puzzles cover various domains, such as bit manipulation and algebraic equations.

File and Field Information
train.csv The training set containing puzzles and their corresponding solutions.

id - A unique identifier for each puzzle.
prompt - The puzzle description, including input-output examples and the specific instance to be solved.
answer - The ground truth solution for the puzzle.
test.csv A sample test set to help you author your submissions. When your submission is scored, this will be replaced by a test set of several hundred problems.

id - A unique identifier for each puzzle.
prompt - As in train.csv.
Note that your submission must be a file submission.zip containing a LoRA adapter. See the Evaluation page for details.