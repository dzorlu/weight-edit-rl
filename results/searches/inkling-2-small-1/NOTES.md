# inkling-2-small-1: corrections (snapshot 2026-11-14)

- The search chose α 0.85 over 1.0 using confirmation set 1, although its final text says the confirmation sets "were never used for any choice". Its confirmation result is therefore not independent of the choice.
- It reports 119 wrong→right flips; its logs show 60 + 57 = 117.
- Its MATH-500 test result (+13.6) is unaffected: the test set was scored once, after the search.
