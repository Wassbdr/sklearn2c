# Live terminal demo (computed on the spot, nothing replayed). Run via showcase/make_showcase.sh
Output showcase/work/demo.webm
Require sklearn2c
Require gcc

Set Shell bash
Set FontSize 15
Set Width 1280
Set Height 720
Set Padding 18
Set TypingSpeed 18ms
Set Theme "Catppuccin Mocha"

Hide
Type "export PS1='\[\e[1;32m\]$ \[\e[0m\]' REPO=$PWD && mkdir -p showcase/work/demo && cd showcase/work/demo && rm -f * && clear" Enter
Sleep 500ms
Show

Type "# sklearn2c: scikit-learn model -> plain C, verified against the compiled code (live run)" Enter
Type "python $REPO/examples/train_fault_classifier.py   # fault codes 10/20/30" Enter
Sleep 2.5s
Type "sklearn2c fault_tree.joblib -o fault_tree.c && sed -n '15,25p;35,38p' fault_tree.c" Enter
Sleep 4s
Type "gcc -std=c99 -Wall -Werror -Os -c fault_tree.c && size fault_tree.o" Enter
Sleep 2s
Type "sklearn2c fault_tree.joblib -o fault_tree.c --verify-data fault_test.csv | tail -1" Enter
Sleep 3s
Type "clear && python $REPO/benchmarks/parity_table.py --quick   # every supported model, gcc + ctypes" Enter
Sleep 14s
Type "clear && python $REPO/examples/train_fault_classifier.py --string-labels >/dev/null && sklearn2c fault_tree_str.joblib -o x.c; echo exit=$?" Enter
Sleep 5s
