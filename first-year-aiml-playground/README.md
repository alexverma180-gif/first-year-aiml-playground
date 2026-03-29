# First-Year AIML Playground

A beginner-friendly repo with:
- 🌸 A Streamlit Iris classifier
- 🧮 ML from scratch (no scikit-learn)
- 🧹 Data cleaning notebook + tiny dataset
- ✅ Tests for confidence

## Run the app
```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## Run tests
```bash
pytest -q
```

## Performance Profiling
To profile the application and identify performance bottlenecks, run the `profile_app.py` script from the repository root:
```bash
python3 profile_app.py
```
This will run the Streamlit app under the `cProfile` profiler and save the output to a file named `profile_output.pstats` in the `first-year-aiml-playground` directory.

To analyze the output, you can use Python's built-in `pstats` module. For example, to print the top 10 functions sorted by cumulative time, you can run the following command:
```bash
python3 -m pstats first-year-aiml-playground/profile_output.pstats
```
Then, within the `pstats` interactive browser, type `sort cumtime` and then `stats 10`.

## Learn
- `scratch_ml/linear_regression.py` – gradient descent from scratch
- `notebooks/data_cleaning_basics.ipynb` – Pandas cleaning & EDA
