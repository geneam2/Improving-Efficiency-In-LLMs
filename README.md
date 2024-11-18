1. Create the Python virtual environment

```
pip install --upgrade pip
python -m venv ./venv
source ./venv/bin/activate # Linux/MacOS
```

2. Install the lm-evaluation-harness submodule

```
git clone --recurse-submodules https://github.com/EleutherAI/lm-evaluation-harness.git
```

3. Install requirements from the lm-evaluation-harness

```
cd lm-evaluation-harness
pip install e .
```
