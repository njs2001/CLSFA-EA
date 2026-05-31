# CLEFA-EA: Entity Alignment Model Based on Contrastive Learning and Sentiment Orientation Assistance

This repository contains the code and supplementary materials for the paper "Entity Alignment Model Based on Contrastive Learning and Sentiment Orientation Assistance".

## Repository Contents
| File Name | Description |
|--------|------|
| `CLEFA-EA.py` | Main implementation of the CLEFA-EA model |
| `Get sentiment.py` | Tool to generate relation emotion mapping files |
| `DBP15k Fr-En.rar` | DBP15K French-English dataset (compressed) |
| `dbp_fr_en_translation.json` | Entity name translation results |
| `Experimental results on hyperparameters...` | Detailed hyperparameter tuning results |
| `requirements.txt` | Specification of dependencies |
| `utilstorch.py` | Utility script for data loading and evaluation metrics |

## Required Downloads (Not Included)
The following files are too large to host on GitHub. Please download them separately:

1.  **Multilingual Sentiment Analysis Model**: tabularisai/multilingual-sentiment-analysis from Hugging Face Hub
2.  **GloVe Pre-trained Word Vectors**: glove.6B.300d.txt
3.  **Translation Service**: We use Baidu General Translation API for entity name translation.

## Quick Start
**This repository uses the DBP15K fr-en dataset as an example.**
1.  Extract the dataset：unrar `DBP15k Fr-En.rar`
2.  Modify the file paths in both `CLEFA-EA.py` and `Get sentiment.py`, to match your local environment
3.  Generate the relation emotion mapping files：Run `python "Get sentiment.py"`
4.  Train and evaluate the model：Run `python CLEFA-EA.py` 
   
## Notes
- GPU is highly recommended for faster training
