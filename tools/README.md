- Calculating WarpError and CLIPScore

CAUTION: All quantitative metrics like WarpError and CLIPScore are reference-only. Users shall not treat them as golden standards to indiate the quality of the generated videos. 

Please first prepare the data by downloading all the results from: [Google cloud](https://drive.google.com/drive/folders/1YPZchMNZ25aNByd-18NUPj0iZrxe5W_-).

```bash
# Please change the 'method_name' in warp_error.py/clip_score.py to calculate different methods
python warp_error.py
python clip_score.py
```
