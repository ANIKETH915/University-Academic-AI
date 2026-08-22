from rag.ocr_layout import reconstruct_questions_from_layout
from tests.test_ocr_layout_reconstruction import L

lines = [
    L("Duration: 3hrs [Max Marks: 80]", 150, 20),
    L("N.B.: (1) Question No 1 is Compulsory.", 151, 50),
    L("(2) Attempt any three questions out of the remaining five.", 226, 80),
    L("Attempt any FOUR [20]", 243, 120),
    L("What is a process control block?", 236, 150),
    L("acre", 199, 165),
    L("Explain context switching overhead.", 236, 180),
    L("Describe a semaphore with an example.", 236, 210),
    L("What is thrashing in virtual memory?", 236, 240),
    L("Explain paging hardware briefly.", 236, 270),
    L("Explain deadlock detection algorithms in detail. [10]", 236, 320),
    L("Discuss challenges in CPU scheduling. [10]", 236, 350),
    L("Consider the following page reference string [10]", 236, 400),
    L("1 2 3 2 1 4", 313, 420),
    L("Compute the number of faults using LRU.", 236, 440),
    L("What are five types of page replacement? 10]", 236, 480),
    L("Explain the banker's algorithm with an example. 10]", 236, 510),
    L("Explain two-phase locking in detail. 10]", 236, 540),
    L("Describe wait-for graphs. 10]", 236, 570),
    L("Explain RAID levels with an example. 10]", 236, 600),
    L("Explain journaled file systems. 10]", 236, 630),
    L("Explain indexed allocation in detail. 10]", 236, 660),
]
text = reconstruct_questions_from_layout(lines)
print(text)
print("N", len(text.splitlines()))
