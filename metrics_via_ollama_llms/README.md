# metrics_test_repo/metrics_via_ollama_llms
This branch has the more consolidated scripts.

They measure:

Context Precision

Context Recall

Faithfulness

According to the RAGAS implementation.

To use:
```python
python3 ./faithfulness_silent.py --models-file ../data/models_file4 --data-file ../data/scenarios4.json 
```
where the name "faithfulness" changes according to the measure you want.
As input in the data folder you need:
- list of models to be tested and
- scenarios

Both files are in json format.
