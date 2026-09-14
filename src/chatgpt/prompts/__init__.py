"""Prompt templates, split by experiment role.

    main_prompts.py -> MAIN treatments (Direct without CoT, Code-Based)
    cot_prompts.py  -> SECONDARY treatment (Direct + Zero-Shot-CoT)

The split is structural on purpose: a main-experiment runner imports
main_prompts and therefore cannot accidentally route to a CoT prompt.
"""
