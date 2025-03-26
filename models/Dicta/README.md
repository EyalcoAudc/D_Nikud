---
license: cc-by-4.0
language:
- he
inference: false
---
# DictaBERT-large-char-menaked: An open-source BERT-based model for adding diacritiziation marks ("nikud") to Hebrew texts

This model is a fine-tuned version of [DictaBERT-large-char](https://huggingface.co/dicta-il/dictabert-large-char), dedicated to the task of adding nikud (diacritics) to Hebrew text. 

The model was trained on a corpus of modern Hebrew texts manually diacritized by linguistic experts. 
As of 2025-03, this model provides SOTA performance on all modern Hebrew vocalization benchmarks as compared to all other open-source alternatives, as well as when compared with commercial generative LLMs.

Note: this model is trained to handle a wide variety of genres of modern Hebrew prose. However, it is not intended for earlier layers of Hebrew (e.g. Biblical, Rabbinic, Premodern), nor for poetic texts.

Sample usage:

```python
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('dicta-il/dictabert-large-char-menaked')
model = AutoModel.from_pretrained('dicta-il/dictabert-large-char-menaked', trust_remote_code=True)

model.eval()

sentence = 'בשנת 1948 השלים אפרים קישון את לימודיו בפיסול מתכת ובתולדות האמנות והחל לפרסם מאמרים הומוריסטיים'
print(model.predict([sentence], tokenizer))
```

Output:
```json
['בִּשְׁנַת 1948 הִשְׁלִים אֶפְרַיִם קִישׁוֹן אֶת לִמּוּדָיו בְּפִסּוּל מַתֶּכֶת וּבְתוֹלְדוֹת הָאׇמָּנוּת וְהֵחֵל לְפַרְסֵם מַאֲמָרִים הוּמוֹרִיסְטִיִּים']
```

### Matres Lectionis (אימות קריאה)

As can be seen, the predict method automatically removed all the matres-lectionis (אימות קריאה). If you wish to keep them in, you can specify that to the predict function:

```python
print(model.predict([sentence], tokenizer, mark_matres_lectionis = '*'))
```

Output:

```json
['בִּשְׁנַת 1948 הִשְׁלִים אֶפְרַיִם קִישׁוֹן אֶת לִי*מּוּדָיו בְּפִי*סּוּל מַתֶּכֶת וּבְתוֹלְדוֹת הָאׇמָּנוּת וְהֵחֵל לְפַרְסֵם מַאֲמָרִים הוּמוֹרִיסְטִיִּים']
```


## License

Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg





