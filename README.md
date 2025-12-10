# uma-trainer
umazing! a training optimizer for umamusume: pretty derby
- a greedy auto-runner using heuristics, specifically NorthernLion's method for training.
- spark farming using template matching and OCR
- plays "good enough" to run for you. turn it on and leave

## Roadmap

### Completed
- template matching for UI detection
- card detection in training regions
- detect support level
- detect unity training + spirit burst
- mouse/keyboard control with pyautogui
- decision making calculations (primitive)

### To-do
- event choice (requires databse. C# / EF Core...?)
- race planning
- database population
- refactor so not everything is stuffed in main
- scale to resolutions outside of 16:9 (evenutally)

#### Ace Trainer (maybe)
- ace trainer mode with stricter stat targets
- detecting Riko Kashimoto
- web ui for configuration (target stats, scenarios, support deck)

#### Complete Simulator (probably not)
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
- MAKE SURE THAT IT WORKS PROPERLY!! i dont want to spend time debugging it

don't be the reason why i have to add more rules