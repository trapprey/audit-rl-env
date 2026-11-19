TRAIN = [
    (1001, {"preset": "clean"}),
    (1002, {"preset": "standard"}),
    (1003, {"preset": "single"}),
    (1004, {"preset": "interacting"}),
    (1005, {"preset": "standard"}),
    (1006, {"preset": "heavy"}),
    (1007, {"preset": "clean"}),
    (1008, {"preset": "quarter"}),
    (1009, {"preset": "interacting"}),
    (1010, {"preset": "standard", "ledger_order": "shuffled"}),
]

DEV = [
    (2001, {"preset": "clean"}),
    (2002, {"preset": "single"}),
    (2003, {"preset": "standard"}),
    (2004, {"preset": "interacting"}),
    (2005, {"preset": "heavy"}),
    (2006, {"preset": "clean"}),
    (2007, {"preset": "standard"}),
    (2008, {"preset": "quarter"}),
    (2009, {"preset": "interacting"}),
    (2010, {"preset": "standard", "ledger_order": "shuffled"}),
]

SPLITS = {"train": TRAIN, "dev": DEV}
