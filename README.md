# Fuzzy Linear Temporal Logic on Finite traces
The repository implements a fast conformance checker for fuzzy logs against temporal specifications in Fuzzy Linear Temporal Logic on finite traces (FLTLf) as described in the paper "Conformance Checking of Fuzzy Logs against Declarative Temporal Specifications".

## Requirements
Implemented with Python 3.8.15, the project requires with the following dependencies:
```
torch==1.13.1
lark==1.1.9
```

## Usage
Once the requirements have been installed, a simple fuzzy log can be defined as a list of lists where each trace is a list of events, each event is a list of predicate values. Use the value 0 for predicates that were not logged in an event
```
traces = []
traces.append( [[0.11, 0.12, 0.13, 0.14], [0.21, 0.22, 0.23, 0.24], [0.31, 0.32,  0.33, 0.34,], [0.41, 0.42, 0.43, 0.44]] ) 
traces.append( [[0.8, 0.3, 0.1, 0.9], [0.9, 0.9, 0.93, 0.94], [0.91, 0.03, 0.4, 0.72]] )
traces.append( [[0.84, 0.3, 0.1, 0.23], [0.9, 0.07, 0.4, 0.14], [0.21, 0.93, 0.04, 0.82], [0.51, 0.03, 0.32, 0.72], [0.63, 0.03, 0.41, 0.92], [0.12, 0.63, 0.03, 0.41]] )
traces.append( [[1, 1, 1, 1]] )
traces.append( [[0, 0, 0, 0], [0.51, 0.32, 0.89, 0.72]] )
```
this fuzzy event log contains 5 traces, the maximum number of events is 6 and there are 4 predicates. Predicates must be declared with a Python list
```
predicate_names = ["cobot_holds", "human_holds", "human_glues", "quality_checking"]
```
The log is then converted into a PyTorch tensor with
```
converter: Converter = Converter(predicate_names, traces)
tensor_log: torch.Tensor = converter.log2tensor(verbose=False)
max_t = converter.max_t
batch_size = converter.batch_size
```
where `tensor_log.shape` is equal to `(5, 6, 4)`. The next step is the definition of an LTLf specification and its parsing
```
string = "G((cobot_holds & human_holds) => X(human_glues))"
parser = LTLfParser(predicate_names, tensor_log, max_t, batch_size)
formula = parser(string)
```
the compliance of the fuzzy log against the LTLf formula can be checked with
```
print(formula.eval(i))
```
This code can be found in the `demo.py` file. For reproducing the results in the BPM 2024 paper, just run `python run_experiments.py`. The results will be save in the `results` folder.

## A Fast Crisp LTLf Checker
The implementation uses the Zadeh t-norm and co-norm that are the fuzzy counterparts of the AND and OR symbol in propositional logic. Therefore, the code can be used as a fast checker for LTLf where the truth values of the event log are just 0 or 1.

## LTLf Syntax
The LTLf syntax adopted is the following:
|        Symbol  |            Meaning            |Example          |
|----------------|-------------------------------|-----------------|
|[a-z][a-z0-9_]* |propositions                   |`cobot_holds`    |
|true            |True                           |`true`           |
|false           |False                          |`false`          |
|&, &&           |And                            |`a && b`         |
|\|, \|\|        |Or                             |`a \|\| b`         |
|!, ~            |Not                            |`!a`             |
|->, =>          |Implication                    |`a => b`         |
|X               |Next                           |`X(a)` or `X a`  |
|wX              |Weak Next                      |`wX(a)` or `wX a`|
|U               |Until                          |`a U b`          |
|W               |Weak Until                     |`a W b`          |
|R               |Release                        |`a R b`          |
|F               |Finally                        |`F(a)`           |
|G               |Globally                       |`G(a)` or `G a`  |
|                | ||

## Reference
Please use the following bibtex entry if you use this code in your work
```

```
