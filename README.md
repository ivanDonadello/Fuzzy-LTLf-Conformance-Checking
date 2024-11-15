# Fuzzy Linear Temporal Logic on Finite traces
The repository implements a fast conformance checker for fuzzy logs against temporal specifications in Fuzzy Linear Temporal Logic on finite traces (FLTLf) as described in the paper "Conformance Checking of Fuzzy Logs against Declarative Temporal Specifications".

## Requirements
Implemented with Python 3.8.15, the project requires with the following dependencies:
```
torch==1.13.1
lark==1.1.9
pandas==2.2.3
```

## Usage
Once the requirements have been installed, a simple fuzzy log can be defined as a list of lists where each trace is a list of events, each event is a list of predicate values. Use the value 0 for predicates that were not logged in an event
```
traces = []
traces.append( [[0.11, 0.12, 0.13, 0.14], [0.21, 0.22, 0.23, 0.24], [0.31, 0.32,  0.33, 0.34,], [0.41, 0.42, 0.43, 0.44]] ) 
traces.append( [[0.08, 0.3, 0.1, 0.9], [0.9, 0.9, 0.93, 0.94], [0.91, 0.03, 0.4, 0.72]] )
traces.append( [[0.14, 0.3, 0.4, 0.23], [0.2, 0.07, 0.4, 0.14], [0.2, 0.93, 0.7, 0.82], [0.31, 0.03, 0.42, 0.72], [0.33, 0.03, 0.41, 0.92], [0.12, 0.63, 0.03, 0.41]] )
traces.append( [[0.2, 1, 0.3, 1], [0.6, 1, 0.43, 1], [0.6, 1, 0.4, 1]] )
traces.append( [[0.3, 0, 0, 0], [0.51, 0.32, 0.89, 0.72]] )
```
this fuzzy event log contains 5 traces, the maximum number of events is 6 and there are 4 predicates. Predicates must be declared with a Python list
```
predicate_names = ["cobot_holds", "human_holds", "human_glues", "quality_checking"]
```
The log is then converted into a PyTorch tensor with
```
converter = Converter(predicate_names)    
tensor_log = converter.log2tensor(tensor_log,verbose, skippadding) 

core.tensor_log, input.predicate_names = converter.slice_tensor_log(tensor_log, formula, verbose)
core.tensor_log = core.tensor_log.to(core.device)

core.batch_size = converter.batch_size
core.maxlength = converter.maxlength
```
The next step is the definition of an LTLf specification and its parsing
```
parser = LTLfParser()
formula = "(F(cobot_holds)) > 0.5" 
pyformula = parser(formula)
```
the compliance of the fuzzy log against the LTLf formula can be checked with
```
i = 0
visitor = core.Visitor()
visitor.visit(pyformula, i)
```
This code can be found in the `demo.py` file. For reproducing the results of the experiments, just run `python simple_exp.py` and `python DECLARE_exp.py`. The results will be save in the `results` folder.

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
|\|, \|\|        |Or                             |`a \|\| b`       |
|!, ~            |Not                            |`!a`             |
|->, =>          |Implication                    |`a => b`         |
|X               |Next                           |`X(a)`           |
|WX              |Weak Next                      |`WX(a)`          |
|U               |Until                          |`aUb`            |
|W               |Weak Until                     |`aWb`            |
|R               |Release                        |`aRb`            |
|M               |Strong Release                 |`aMb`            |
|F               |Finally                        |`F(a)`           |
|G               |Globally                       |`G(a)`           |
|<,<=,>,>=,!=,== |Comparison operators           |`((a) >= 0.5)` or `((a) >= (b))`   |

## Reference
Please use the following bibtex entry if you use this code in your work
```
@inproceedings{donadello2024conformance,
  title={Conformance Checking of Fuzzy Logs Against Declarative Temporal Specifications},
  author={Donadello, Ivan and Felli, Paolo and Innes, Craig and Maggi, Fabrizio Maria and Montali, Marco},
  booktitle={International Conference on Business Process Management},
  pages={39--56},
  year={2024},
  organization={Springer}
}
```
