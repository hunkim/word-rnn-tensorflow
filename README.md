# word-rnn-tensorflow
[![Build Status](https://travis-ci.org/hunkim/word-rnn-tensorflow.svg?branch=master)](https://travis-ci.org/hunkim/word-rnn-tensorflow)

Multi-layer Recurrent Neural Networks (LSTM, RNN) for word-level language models in Python using TensorFlow.

Mostly reused code from https://github.com/sherjilozair/char-rnn-tensorflow which was inspired from Andrej Karpathy's [char-rnn](https://github.com/karpathy/char-rnn).

# Requirements
- [Tensorflow](http://www.tensorflow.org)

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

## Word-RNN
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

## Word-RNN (with beam search)
```
KING RICHARD III:
you, by thou and and be not made at London with my legs.

ROMEO:
This is the quarrel; for this fellow: ho?

RATCLIFF:
I pray thee, moralize them.

FERDINAND:
This is fairy gold, boy, or 'twill do more than a man that calls upon me?
reflect I not with the basilisk: I have Forbidden bandying in chance,

GREGORY:
and fret, that I have heard the princess As that she hath been deposed;
Or late: farewell; my lord. Light to my boon.

LADY CAPULET:
I will not speak!

KING RICHARD II:
Marshal, Lartius, thou art,
Commit'st my abilities are no more doublets thanbacks, thou takest,
Within the morning, and my advancement?
I must not be conducted.

DUKE VINCENTIO:
Repent you the time to visit, I am not strangely in my mouth,
those that was germane to him, as thou talk'st of contrary.

KING RICHARD II:
Marshal, Lartius, thou fortunate!

MONTAGUE:
Comfort, my lord; and say you shall be stoned;
but my heart shall have a man as stead me thy appointment,
give me thy brother exercise;

BUCKINGHAM:
I pray you, mark your penitence, if it be violent, As when I would not disdain.
```

## Char-RNN

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
# Projects
If you have any project using this word-rnn, please let us know. I'll list up your project here.

- http://bot.wpoem.com/ (Simple poem generator in Korean)


# Contribution
Your comments (issues) and PRs are always welcome.
