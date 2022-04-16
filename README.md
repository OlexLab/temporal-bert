# temporal-bert
A collection of temporally fine-tuned BERT and Clinical BERT models, and associated code, designed for performing temporal type disambiguation.

The models in this repository were temporally fine-tuned on 2 types of temporal tasks: binary temporal sentence classification and multi-label temporal type classification. Fine-tuning and other details are described in Amy Olex's dissertation titled "Temporal disambiguation of relative temporal expressions in clinical texts using temporally fine-tuned contextual word embeddings." <reference to be added>
  
  All BertBase models are initilized from the original "bert-base-uncased" model described in Devlin et.al. All ClinBioBert models were initilized from the BERT model fine-tuned on biomedical literature and clinical notes by Alsentzer et.al.
  
## References:
  - Devlin J, Chang M-W, Lee K et al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers). Minneapolis, Minnesota: Association for Computational Linguistics, 2019, 4171–86.

  - Alsentzer E, Murphy JR, Boag W et al. Publicly Available Clinical BERT Embeddings. arXiv:190403323 [cs] 2019. https://github.com/EmilyAlsentzer/clinicalBERT

