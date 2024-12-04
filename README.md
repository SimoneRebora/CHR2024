# CHR2024
Scripts for the paper:  
Simone Rebora and Gabriele Vezzani, "Models of Literary Evaluation and Web 2.0. An Annotation Experiment with Goodreads Reviews", *CHR2024* ([PDF](https://2024.computational-humanities-research.org/papers/paper46))  

## Preliminary notes

All ".ipynb" scripts are designed to work with [Google Colab](https://colab.research.google.com/). You will need to connect your Drive to the Notebooks and have all datasets saved in the folder "MyDrive/CHR2024".  
Because of copyright limitations, we cannot share the datasets here. If interested in getting access to them, you can contact us by describing your intended use.  

## Scripts overview

The scripts are linked to different sections of our paper. Here below you can find a brief description and an indication of the section of the paper where the script was used.  
- **Compute_Agreement.ipynb** *(Paper section 3.2).* Script used to calculare inter-annotator agreement.
- **Transformers_kfold.ipynb** *(Paper section 4.1).* Script used to fine-tune three Transformer models on the annotated dataset.
- **Transformers_learning_curve.ipynb** *(Paper section 4.1).* Script used to calculate efficiency of the best-performing model with differing amounts of training materials.
- **Transformers_save_model.ipynb** *(Paper section 4.1).* Script used to save the best-performing model when trained on a selected dataset. The model has been published on [HuggingFace](https://huggingface.co/GVezzani/literary_evaluation_classifier).
- **Evaluate_GPT_prompt_engineering.ipynb** *(Paper section 4.2).* Script used to evaluate the efficiency of different GPT4 prompting strategies on the selected dataset.
