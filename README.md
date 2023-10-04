# CSM: Confidence-based Self-imitation Model

Code and data for our EACL 2023 paper [Self-imitation Learning for Action Generation in Text-based Games](https://aclanthology.org/2023.eacl-main.50.pdf).



## How to Train

1. **Set Up the Environment**
   


   ```bash
   pip install torch==1.5.1 transformers==2.5.1 jericho fasttext wandb importlib_metadata
   python -m spacy download en_core_web_sm
   ```


2. **Download Pretrained Models and Walkthrough Data**

    Please download the pretrained GPT-2 model, which was trained by the [princeton-nlp/calm-textgame](https://github.com/princeton-nlp/calm-textgame), from the following link: [Download GPT-2 Model](https://drive.google.com/file/d/1PBAXq4LW9pdVdLFyF_donwCV46wBX1zD/view).

    Once downloaded, place the model files in the following directory structure:

    ```
    ├── download-models
    │   ├── gpt2
    │   │   ├── config.json
    │   │   ├── merges.txt
    │   │   ├── pytorch_model.bin
    │   │   └── vocab.json
    │   └── jericho_walkthrough_data
    ```

    Please note that the `jericho_walkthrough_data` files is used for validating the accuracy of the language model after fine-tuning and **does not participate in the training process**.


3. **Run the Training Script**

   Initiate training with the following command:

   ```bash
   sh drrn/run_csm.sh
   ```

## Citation
```
@inproceedings{shi-etal-2023-self,
    title = "Self-imitation Learning for Action Generation in Text-based Games",
    author = "Shi, Zijing  and
      Xu, Yunqiu  and
      Fang, Meng  and
      Chen, Ling",
    booktitle = "Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics",
    month = may,
    year = "2023",
    address = "Dubrovnik, Croatia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.eacl-main.50",
    pages = "703--726",
}

```
## Acknowledgements

We thank [princeton-nlp/calm-textgame](https://github.com/princeton-nlp/calm-textgame) for providing the excellent ClubFloyd dataset and CALM codebase.


