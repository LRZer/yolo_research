# Epoch-label correction

The source checkpoint is the 50-epoch B50 run's `last.pt`. An older probe fallback mislabeled `last.pt` as epoch 150 in the raw CSV. Its actual completed epoch is **50**; the Markdown and JSON views in this directory have been corrected. The numeric probe measurements are unchanged. The probe script has also been fixed to read `train_args.epochs`, so future `last.pt` reports do not use the stale fallback.
