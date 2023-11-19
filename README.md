# CS234 (Coding)
Coding solution for assignments in:
https://web.stanford.edu/class/cs234/
</br>
I might be mistaken, so please let me know if there are any errors

# Assignment1
Run deterministic or stochastic policy with `python --env` command.
(Deterministic by default)
```
python assignment1/final/vi_and_pi.py --env Stochastic-4x4-FrozenLake-v0
python assignment1/final/vi_and_pi.py --env Deterministic-4x4-FrozenLake-v0
```

# Assignment2
MinAtar is needed to run models.
```
git clone https://github.com/kenjyoung/MinAtar.git
cd MinAtar
pip install .
```

# Assignment3
Still working on Assignment3.

If you encounter issues with the following command:
```
pip install -r requirements.txt

>> error in gym setup command: 'extras_require' must be a dictionary whose values are strings or lists of strings containing valid project/version requirement specifiers.
```
- Modify `gym==0.21` to `gym`
- Run `pip install -r requirements.txt`
- Revert to the original version by running `pip install gym==0.21`.