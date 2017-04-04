# word-rnn-tensorflow
[![Build Status](https://travis-ci.org/hunkim/word-rnn-tensorflow.svg?branch=master)](https://travis-ci.org/hunkim/word-rnn-tensorflow)

Multi-layer Recurrent Neural Networks (LSTM, RNN) for word-level language models in Python using TensorFlow.

Mostly reused code from https://github.com/sherjilozair/char-rnn-tensorflow which was inspired from Andrej Karpathy's [char-rnn](https://github.com/karpathy/char-rnn).

# Requirements
- [Tensorflow 1.1.0rc0](http://www.tensorflow.org)

# Basic Usage
To train with default parameters on the tinyshakespeare corpus, run:
```bash
python train.py
```

To sample from a trained model
```bash
python sample.py
```

To pick using beam search, use the `--pick` parameter. Beam search can be
further customized using the `--width` parameter, which sets the number of beams
to search with. For example:
```bash
python sample.py --pick 2 --width 4
```

# Sample output

### Word-RNN
```
LEONTES:
Why, my Irish time?
And argue in the lord; the man mad, must be deserved a spirit as drown the warlike Pray him, how seven in.

KING would be made that, methoughts I may married a Lord dishonour
Than thou that be mine kites and sinew for his honour
In reason prettily the sudden night upon all shalt bid him thus again. times than one from mine unaccustom'd sir.

LARTIUS:
O,'tis aediles, fight!
Farewell, it himself have saw.

SLY:
Now gods have their VINCENTIO:
Whipt fearing but first I know you you, hinder truths.

ANGELO:
This are entitle up my dearest state but deliver'd.

DUKE look dissolved: seemeth brands
That He being and
full of toad, they knew me to joy.
```

### Char-RNN
```
ESCALUS:
What is our honours, such a Richard story
Which you mark with bloody been Thilld we'll adverses:
That thou, Aurtructs a greques' great
Jmander may to save it not shif theseen my news
Clisters it take us?
Say the dulterout apy showd. They hance!

AnBESS OF GUCESTER:
Now, glarding far it prick me with this queen.
And if thou met were with revil, sir?

KATHW:
I must not my naturation disery,
And six nor's mighty wind, I fairs, if?

Messenger:
My lank, nobles arms;
```

## Beam search

Beam search differs from the other `--pick` options in that it does not greedily
pick single words; rather, it expands the most promising nodes and keeps a
running score for each beam.

### Word-RNN (with beam search)
```
# python sample.py --prime "KING RICHARD III:" -n 100 --pick 2 --width 4

KING RICHARD III:
you, and and and and have been to be hanged, I am not to be touched?

Provost:
A Bohemian born, for tying his own train,
Forthwith by all that converses more with a crow-keeper;
I have drunk, Broach'd with the acorn cradled. Follow.

FERDINAND:
Who would not be conducted.

BISHOP OF ELY:
If you have been a-bed an acre of barren ground, hath holy;
I warrant, my lord restored of noon.

ISABELLA:
'Save my master and his shortness whisper me to the pedlar;
Money's a medler.
That I will pamper it to complain.

VOLUMNIA:
Indeed, I am
```

### Word-RNN (without beam search)
```
# python sample.py --prime "KING RICHARD III:" -n 100

KING RICHARD III:
marry, so and unto the wind have yours;
And thou Juliet, sir?

JULIET:
Well, wherefore speak your disposition cousin;
May thee flatter.
My hand will answer him;
e not to your Mariana Below these those and take this life,
That stir not light of reason.
The time Lucentio keeps a root from you.
Cursed be his potency,
It was my neighbour till the birth and I drank stay.

MENENIUS:
Here's the matter,
I know take this sour place,
they know allegiance Had made you guilty.
You do her bear comfort him between him or our noble bosom he did Bolingbroke's
```

# Projects
If you have any project using this word-rnn, please let us know. I'll list up your project here.

- http://bot.wpoem.com/ (Simple poem generator in Korean)


# Contribution
Your comments (issues) and PRs are always welcome.
