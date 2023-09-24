# ![icon](/assets/feature.jpg)&nbsp; Tic Tac Transformer

A tiny GPT trained to play tic-tac-toe

## How does it work?

We teach a language model to speak tic-tac-toe

The language is simple - there are 11 tokens

- **0-8**: moves on the board
- **9**: start game
- **10**: pad

The sequence length is 10, so a game always starts with <9> and can at most fill the board

Players take turns

Duplicate moves are illegal

**Example**

seq: [9, 4, 0, 2, 1, 6, 10, 10, 10, 10]

- player 1 puts an X at position 4 (the middle)
- player 2 puts an O at position 3 (top left)
- player 1 puts an X at position 2 (top right)
- player 2 puts an O at position 1 (top middle)
- player 1 puts an X at position 6 (bottom left)
- padding

```
[O] [O] [X]
[ ] [X] [ ]
[X] [ ] [ ]
```

player 1 wins

## Try for yourself

Play the AI!

```bash
python play_ai.py
```

## Training

Generate pre-training data

```bash
python generate_data.py
```

Run pre-training

```bash
python train.py
```

RL fine-tuning

```bash
python reinforcement_learn.py
```

Run benchmark

```bash
python benchmark.py
```
