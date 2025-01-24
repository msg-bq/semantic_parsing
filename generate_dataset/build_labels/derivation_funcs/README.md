# RandomCFGDeriver
A project for Theory of Computability class that takes a context free grammar file and derives a random correct string.

Included files:

Filename | Description
---------|-------------
grammar.py | python3 source containing the grammar class
variable.py | python3 source containing the variable class
variable_reference.py | python3 source containing the variable reference class
random_cfg_deriver.py | the main python3 source that algorithmically determines a random correct string from a given cfg file
teatime.cfg | a file containing a context free grammar I made up.
sampleout | a text file containing a sample output

**How to run:**

python3 random_cfg_deriver.py {cfg filename}

About the project:

A context free grammar is made of a set of rules that describe how a variable can be replaced with a string comprising of symbols and/or more variables.

The CFG(context free grammar) file is a text file with the extension .cfg

The CFG file is organized into lines, with variable names being entirely upper case letters, and alphabet symbols being always lower case.

There are four kinds of lines within the CFG file:

Line Type      | Description
--------------------------| --------------------
1. Whitespace | the line only has whitespace and will be ignored
2. Comment | the line starts with a '#' character and will be ignored
3. First Rule | the line has a form of {VARIABLE}={PRODUCTION} which defines a production rule for a new variable called VARIABLE. The <PRODUCTION> characters include all characters between the '=' and the newline, which may be a mix of upper case letters defining variable names, and lower case letters definint alphabet symbols. The space character is allowed as a symbol.
4. Alternative Rule | the line has a form of \|{PRODUCTION} and must come after a First Rule or an Alternative Rule line. It defines another possible production for the same variable as the preceding rule.

The grammar must include a variable called START which is the start variable.

There is an assumption that in productions, variable names are always separated by at least one symbol.

The random_cfg_deriver program derives strings randomly from a given CFG file by starting with the START variable, and then randomly choosing the next variable to expand. Upon having randomly chosen a variable to expand, the program will randomly choose from the available productions for that variable. The process continues until all variables have been randomly expanded.
