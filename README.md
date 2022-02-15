# LearningByDoing NeurIPS 2021 Competition: Standalone Code and Results

## Learning By Doing!

This is the official implementation of the [***Learning By Doing*** **NeurIPS 2021 Competition**](https://learningbydoingcompetition.github.io/)!

### Why are we releasing this?

- To promote work at the intersection of control theory, reinforcement learning, and causality!
- To allow others to try out new techniques with our competition data!
- To help future competitions!
- To enshrine our prize winners! 🏅

### Quick start!

Once you've gotten everything installed, you can easily replicate the final results of the competition! 😎

Track CHEM 🧑‍🔬: `python run_chem_winners.py`

| Team | Score | Rank |
|---|:---:|:---|
| Ajoo | 0.0890 | 1 🏆 | 
| TeamQ | 0.3385 | 2 |
| GuineaPig | 0.3386 | 3 |

Track ROBO 🤖: `python run_robo_winners.py`

| Team | Score | Rank |
|---|:---:|:---|
| Ajoo | 0.918 | 1 🏆 | 
| TeamQ | 16.121 | 2 |
| Factored.ai (jmunozb) | 29.539 | 3 |

Congratulations to our competition prize winners! We also thank teams Ajoo, TeamQ, and KC988 for discussing their solutions at our NeurIPS 2021 session!

### What does this repo contain?

**Track CHEM** 🧑‍🔬
| Feature | Status |
|---|:---:|
| Competition systems | ✅ |
| Final phase evaluation | ✅ |
| Prize winner submissions | ✅ |
| Starter kit | ✅ |
| Training data | ✅ |
| Training data generation script | ✅ |
| Prize winner implementations | 🔗 |
| Intermediate phase evaluation | ❌ |
| Codalab bundle infrastructure | ❌ |
| Submission evaluation using Docker | ❌ |

**Track ROBO** 🤖
| Feature | Status |
|---|:---:|
| Competition systems | ✅ |
| Final phase evaluation | ✅ |
| Prize winner controllers | ✅ |
| Starter controller | ✅ |
| Zero baseline controller | ✅ |
| Training trajectories | ✅ |
| Target trajectories | ✅ |
| Oracle controller | ✅ |
| Training trajectories generation script | ✅ |
| Target trajectories generation script | ✅ |
| Intermediate phase evaluation | ❌ |
| Codalab bundle infrastructure | ❌ |
| Controller evaluation using Docker | ❌ |

Legend:
- ✅: released!
- 🔗: available at link below
- ⏳: not yet available -- currently in progress
- ❌: not planned (but please contact us if you need it!)

### Have any suggestions? Tell us!

We'd love to hear your feedback for any ways we can improve upon our ideas for future competitions at the intersection of control theory, reinforcement learning, and causality!

Tell us here: `LearningByDoing AT math DOT ku DOT dk`

Thank you, and keep *Learning By Doing*!

---

# Installation Instructions

This implementation was tested using Python3.7. Different versions of Python3 should work; please let us know if you have difficulty. The `setup.py` file specifies the same version of the libraries used on Codalab.

## (Optional) Create and source virtualenv

Optionally, first create a Python3 virtualenv, to which the LBD package will be installed.
These instructions use a default virtualenv directory of `~/envs/`.

```
virtualenv -p /usr/bin/python3 ~/envs/lbd
source ~/envs/lbd/bin/activate
```

## Clone and install repo

```
git clone git@github.com:LearningByDoingCompetition/learningbydoing-comp-staging.git
cd learningbydoing-comp-staging/
pip install -e .
```

## Tests

To run the tests for this please, run the following:

```
pytest tests/
```

# Final Round Results

## Track CHEM

To run the final phase of Track CHEM, run the following:

```
python run_chem_winners.py
```

Example output (evaluation time may differ):

```
LearningByDoing: Track CHEM Final Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rank 1: Ajoo
100%|██████████████████████████████████████████████████████████████| 12/12 [00:49<00:00,  4.14s/it]
Results:
Loss: 0.0890
Duration (seconds): 49.74

Rank 2: TeamQ
100%|██████████████████████████████████████████████████████████████| 12/12 [00:46<00:00,  3.86s/it]
Results:
Loss: 0.3385
Duration (seconds): 46.33

Rank 3: GuineaPig
100%|██████████████████████████████████████████████████████████████| 12/12 [00:45<00:00,  3.76s/it]
Results:
Loss: 0.3386
Duration (seconds): 45.17

```

## Track ROBO

To run the final phase of Track ROBO, run the following:

```
python run_robo_winners.py
```

Example output (evaluation time may differ):

```
LearningByDoing: Track ROBO Final Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rank 1: Ajoo
100%|████████████████████████████████████████████████████████████| 240/240 [01:17<00:00,  3.12it/s]
Loss: 0.918
Controller-Runtime: 28.4

Rank 2: TeamQ
100%|████████████████████████████████████████████████████████████| 240/240 [02:30<00:00,  1.60it/s]
Loss: 16.121
Controller-Runtime: 87.5

Rank 3: jmunozb
100%|████████████████████████████████████████████████████████████| 240/240 [02:23<00:00,  1.68it/s]
Loss: 29.539
Controller-Runtime: 51.9

```

# Examples

## Track CHEM: Evaluating a Submission

Submissions for Track CHEM are evaluated using the `scripts/run_chem_evaluation.py` script. The following example evaluates Team Ajoo's CHEM submission:

```
python scripts/run_chem_evaluation.py -i data/CHEM/prize_winners/1_Ajoo/ -v
```

Example output (evaluation time may differ):

```
100%|██████████████████████████████████████████████████████████████| 12/12 [00:50<00:00,  4.19s/it]
Results:
Loss: 0.0890
Duration (seconds): 50.32
```

## Track CHEM: Generating Training Data

Training data are generated for Track CHEM using the `scripts/gen_chem_training_data.py` script. For example, the following command generates and saves training data to the directory `output/chem_train/` as a zip file (`-z`):

```
python scripts/gen_chem_training_data.py -o output/chem_train/ -z
```

## Track ROBO: Running Controllers

Controllers for Track ROBO are evaluated using the `scripts/run_robo_controller.py` script. The following example evaluates the starter kit controller:

```
python scripts/run_robo_controller.py -i data/ROBO/controllers/starter_kit/controller/ -t data/ROBO/target_trajectories/
```

Example output (evaluation time may differ):

```
100%|████████████████████████████████████████████████████████████| 240/240 [01:32<00:00,  2.60it/s]
Loss: 67.156
Controller-Runtime: 32.5
```

## Track ROBO: Generating Training Data

Training data are generated for Track ROBO using the `scripts/gen_robo_training_data.py` script. For example, the following command generates and saves training trajectories to the directory `output/robo_train/` while saving debug animations (`-d`) to `output/robo_train/debug/`:

```
python scripts/gen_robo_training_data.py -o output/robo_train/ -d
```

The script will default to 50 trajectories per robot system, although this setting can be changed through the `-n` argument.

Debug animations are informative but increase script runtime. The `-d` argument is optional and thus doesn't need to be specified in order to generate the training data more quickly.

## Track ROBO: Generating Test Data

Track ROBO's test data are generated using the `scripts/gen_robo_test_data.py` script. The following example generates and saves target trajectories to the directory `output/robo_test/` while saving debug animations (`-d`) to `output/robo_test/debug/`:

```
python scripts/gen_robo_test_data.py -o output/robo_test/ -d
```

The target trajectories are a mix of curved trajectories, straight-line traversals, circular arcs, rectangular trajectories, and the letters "L", "B", and "D" that spell out the competition!

Similar to the training data script, the debug argument (`-d`) is optional and can be removed to generate the target trajectories more quickly.

# Prize Winner Implementations

This repository contains the prize winner solutions. For Track CHEM, these are the csv submission files. For Track ROBO, the submissions are controllers.

Below are the implementations from our prize winners. Specifically, implementations for Track CHEM winners are available below, as our repository only contains the csv submissions.

Ajoo:
- https://github.com/Ajoo/lbd-neurips2021

TeamQ:
- https://github.com/Quarticai/learning_by_doing_solution

Factored.ai (jmunozb):
- https://github.com/factoredai/learn-by-doing-NeurIPS-2021

GuineaPig:
- https://github.com/bartbussmann/learningbydoing/blob/main/CHEM_starter_kit/submission.py

# License

The Learning By Doing NeurIPS 2021 Competition Standalone Code is licensed under the GNU Affero General Public License v3.0 (AGPLv3). Please see the license described in the `LICENSE` file for more information.

# Citation

```
@misc{weichwald2022learning,
      title={{Learning by Doing: Controlling a Dynamical System using Causality, Control, and Reinforcement Learning}}, 
      author={Sebastian Weichwald and Søren Wengel Mogensen and Tabitha Edith Lee and Dominik Baumann and Oliver Kroemer and Isabelle Guyon and Sebastian Trimpe and Jonas Peters and Niklas Pfister},
      year={2022},
      eprint={2202.06052},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
