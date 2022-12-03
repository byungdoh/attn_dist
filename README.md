# Entropy- and Distance-Based Predictors From GPT-2 Attention Patterns Predict Reading Times Over and Above GPT-2 Surprisal

## Introduction
This is the code repository for the paper [Entropy- and Distance-Based Predictors From GPT-2 Attention Patterns Predict Reading Times Over and Above GPT-2 Surprisal](https://byungdoh.github.io/pdf/emnlp22_attndist.pdf), including a modified version of the [HuggingFace Transformers](https://huggingface.co/docs/transformers/index) repository for calculating by-word predictors from the attention patterns of the GPT-2 language model. This repository also includes predictors calculated on the Natural Stories (Futrell et al., 2021) and Dundee (Kennedy et al., 2003) corpora, which were used in this work.

## Setup
1) Install the following major dependencies:
- [PyTorch](https://pytorch.org)
- [scipy](https://scipy.org)
2) Install the modified version of the Transformers repository using these commands:
```
cd huggingface
pip install -e .
```

## Predictor Calculation
The command `python main.py INPUT_FILE gpt2 > OUTPUT_FILE` (e.g. `python main.py my_stimuli.sentitems gpt2 > my_stimuli.gpt2_attn.predictors`) can be used to calculate by-word predictors using the modified version of the pre-trained GPT-2.
The input file is split according to `!ARTICLE` delimiters and assigned to different batches. 

```
$ head my_stimuli.sentitems
!ARTICLE
Hello, welcome to our repository.
Please refer to the instructions we've provided in the readme.
```

The output should be a space-delimited 13-column file containing the word and the four predictors that were defined in this work (i.e. `NAE`, `$\Delta$NAE`, `MD`, `EMD`) calculated under three different attention formulations (i.e. `Attn-W`, `Attn-N`, `AttnRL-N`).

```
$ head my_stimuli.gpt2_attn.predictors
word normentw normentn normentr deltaentw deltaentn deltaentr manhattanw manhattann manhattanr emdw emdn emdr
Hello, 5.927118896157481 8.794507116312161 9.307190814986825 nan nan nan 7.571024660021067 16.969935044646263 34.1797798871994 3.354933441275909 7.713587284057249 13.564050507597461
welcome 5.451154500246048 9.211145758628845 9.216887712478638 1.7190166426589713 2.247386842733249 1.9863234367221594 2.936075061559677 5.479778558015823 14.8632253408432 0.8277105334141162 1.3218689802956076 2.820370045940535
to 5.763344496488571 9.207774043083191 9.243371188640594 1.1843443810939789 1.2228091955184937 1.3117113709449768 3.5548617616295815 5.597584202885628 15.53438812494278 1.0099715000574114 1.4059543367982932 2.1809179073091056
our 5.528188034892082 9.308584988117218 9.221545338630676 1.2105535715818405 1.3166876435279846 1.3001386523246765 3.314097225666046 5.39351099729538 14.918231129646301 0.7524056390781805 1.3134454621825413 1.83952708928097
repository. 13.009766221046448 19.555642783641815 19.537510097026825 3.1465869396924973 2.1290265321731567 2.0371715426445007 9.725593976676464 15.033280581235886 28.69909644126892 2.6021767805071088 3.7751427576015204 3.2385816566406875
Please 4.23738856613636 7.894982814788818 7.746921867132187 2.8991691321134567 2.2583371996879578 2.440116971731186 6.370759829878807 9.637452363967896 16.86677050590515 1.5774318103928755 2.2075844971463097 1.6496844133269466
refer 4.657134037464857 8.248523235321045 8.242960751056671 1.0733316130936146 2.0905174016952515 1.9984122812747955 4.143049996346235 7.308557391166687 16.40018081665039 0.7422238288319337 1.0592085798903383 1.2231667492575446
to 6.991287887096405 9.182535290718079 9.318768918514252 2.4268000908195972 1.8814265727996826 1.8793036937713623 6.963574334979057 8.693524271249771 15.798975944519043 1.8530369188524434 1.859701967810536 1.4437423097904238
the 6.445421829819679 9.502938449382782 9.418984770774841 1.0398754626512527 0.9832711815834045 0.9205464720726013 4.728325434029102 6.052533328533173 13.423013389110565 0.9166567891146506 0.9681384636356622 1.0352900580678315
```

## Predictors on Reading Time Corpora
The `data` directory contains the predictors calculated on the Natural Stories self-paced reading corpus and the Dundee eye-tracking corpus, which were used in the regression experiments.

## Additional Remarks
1) The by-word predictors are calculated by *aggregating* the by-head predictors from the *topmost* layer of GPT-2. To examine the individual by-head predictors, the `calc_*` functions can be edited to maintain the individual by-head predictors. For computational efficiency, the `Attn-N` and `AttnRL-N` representations are only calculated at the topmost layer. This can be modified in Line 920 of `huggingface/src/transformers/models/gpt2/modeling_gpt2.py`:
```
# currently hacked to output transforms at last layer only
output_transforms = (i == self.config.num_hidden_layers-1)
```
2) Although it hasn't been tested, the larger variants of GPT-2 (i.e. `gpt2-medium`, `gpt2-large`, `gpt2-xl`) should also be supported. However, this may be computationally infeasible for long input sequences.
3) The modified GPT-2 model returns the raw `Attn-N` and `AttnRL-N` representations as `projected_states` and `resln_states` respectively, which can be further examined.

## Questions
For questions or concerns, please contact Byung-Doh Oh ([oh.531@osu.edu](mailto:oh.531@osu.edu)).