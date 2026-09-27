"""BIS central-banker speeches: historical load, updates, provenance NER.

    jobs start cb_speeches --start 1996-01-01 --end 2026-12-31 [--partition-freq Y]
    jobs update <cb_speeches id>                 new / revised / removed speeches
    jobs start cb_speech_ner --speeches-id <id> --provider ollama --model <name>
    jobs update <cb_speech_ner id>               NER for speeches not in the table yet

    from cb_speeches.speeches import read_speeches
    from cb_speeches.ner import speeches_with_ner
"""
