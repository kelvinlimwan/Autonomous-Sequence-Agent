# Autonmous Sequence Agent

Assignment for AI Planning for Autonomy (COMP90054). Various algorithms are analysed and thoroughly evaluated to design an autonomous player for the modified board game Sequence.

### Guide

The python version supported is `python 3.6+`.
All required classes and methods to implement a new game are defined in the [template.py](template.py)

Only additional import required is the `func_timeout` module. Install with: pip3 install func_timeout

There are three give general agents that will work with any game under directory [agents/group76](agents/group76): [heuristic.py](agents/group76/heurstic.py), [qLearning.py](agents/group76/qLearning.py) and [deepQ.py](agents/group76/deeQ.py). In order to guide the AI's decision making, the methods implement a heuristic function, Q-Learning and Deep Q-Learning respectively.


### Usage
The game can be run with specified runner. The only difference between runner is the first two line of the code (importing different game files.) The options can be found using following command:
```
python sequence_runner.py -h
```
If running Sequence, note that the game will start in fullscreen mode. Press F11 to toggle fullscreen. The game's activity log now appears as a separate window.

### Feature
- save the print as log
- save replay
- play saved replay file
- run multiple games in sequential


### Current Games:
- Sequence


### Limitation
For teams to import their customized python file, they will have to add their team name into the path. For example:
```
import agents.staff_team_random.timeout
```
