1. Create the Python virtual environment

```
pip install --upgrade pip
python -m venv ./venv
source ./venv/bin/activate # Linux/MacOS
pip install -r requirements.txt
```

2. Install the lm-evaluation-harness submodule

```
git clone --recurse-submodules https://github.com/EleutherAI/lm-evaluation-harness.git
```
