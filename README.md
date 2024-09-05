# leaffliction

## How to use

```bash
# Download image dataset and generate distribution chart image
python 01.Distribution.py apple grape

# Augment unbalanced image dataset
python 02.Augmentation.py

# Save transformed image plot
python 03.Transformation.py -src [SRC_PATH] -dst [DST_PATH]
```

## Tensorboard

```bash
tensorboard --logdir runs 
```

## Resources

- [Youtube Coursera CNN](https://www.youtube.com/playlist?list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF)
