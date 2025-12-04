# uma-trainer
umazing! a training optimizer for umamusume: pretty derby
- a greedy auto-runner using heuristics, specifically NorthernLion's method for training.
- spark farming using template matching and OCR
- plays "good enough" to run for you. turn it on and leave

## Roadmap

### Completed
- template matching for UI detection
- card detection in training regions

### To-do
- detect support level
- detect training burst
- decision making calculations
- choice maker for events
- choosing medium in the beginning or end of the run
- mouse/keyboard control with pyautogui
- endgame: unity cup scenario support

#### Ace Trainer (maybe)
- ace trainer mode with stricter stat targets
- detecting Riko Kashimoto
- web ui for configuration (target stats, scenarios, support deck)


#### Complete Simulator (maybe)
- create databases
    - support cards (skills, stats, lb rating. every single card!) - what the whole app will depend on
    - umas (mainly for stat growth, events, required races)
    - race database?
    - scenario database (potentially)

## Contributing
open to contributions, but please follow best practices

- open an issue before major changes
- follow existing code style
- please test your changes. or at the very least explain what you tested

don't be the reason why i have to add more rules