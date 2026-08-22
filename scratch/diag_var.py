"""Diagnose variable-count PDF extraction losses."""
import os
import fitz
from rag.dynamic_ingest import DynamicIngestPipeline
from rag.workspace_db import WorkspaceDB
from rag.vector_store import VectorStore
from rag.question_extractor import extract_questions_from_page_text

TOPICS_POOL = [
    "Convolutional Neural Networks for image classification",
    "Recurrent Neural Networks and Vanishing Gradient Problem",
    "Long Short Term Memory Networks and Gated Recurrent Units",
    "Autoencoders Denoising and Variational Architectures",
    "Generative Adversarial Networks Generator Discriminator Loss",
    "Gradient Descent Optimization Adam Stochastic RMSprop",
    "Backpropagation Error Signal Chain Rule Derivatives",
    "Activation Functions ReLU Sigmoid Softmax Tanh Properties",
    "Regularization Techniques L1 L2 Dropout Weight Decay",
    "Transfer Learning Fine Tuning Pretrained Backbone Models",
]

def create_synthetic_pdf(num_questions: int, filename: str) -> str:
    doc = fitz.open()
    page = doc.new_page()
    lines = [
        "UNIVERSITY EXAMINATION",
        "SUBJECT: ADVANCED ALGORITHMS | SEMESTER 7",
        "Duration: 3 Hours | Max Marks: 80",
        "Instructions: Attempt all questions. Assume suitable data if necessary.",
        ""
    ]
    for q_i in range(1, num_questions + 1):
        topic = TOPICS_POOL[(q_i - 1) % len(TOPICS_POOL)]
        p_num = (q_i - 1) // 5 + 1
        s_char = chr(97 + ((q_i - 1) % 5))
        q_str = f"Q{p_num}({s_char}) Describe architectural principles and implementation of {topic} in computing. [10 Marks]"
        lines.append(q_str)
    page.insert_text((50, 50), "\n".join(lines), fontsize=10)
    os.makedirs("scratch", exist_ok=True)
    pdf_path = os.path.join("scratch", filename)
    doc.save(pdf_path)
    doc.close()
    return pdf_path

path = create_synthetic_pdf(10, "diag_10.pdf")
doc = fitz.open(path)
text = doc[0].get_text()
doc.close()
print("=== RAW PDF TEXT ===")
print(repr(text[:1500]))
print("=== EXTRACT ===")
acc, rej = extract_questions_from_page_text(text, 1, "diag_10.pdf", "ws", year=2024)
print("accepted", len(acc), [q["question_id"] for q in acc])
print("rejected", [(r.get("question_id"), r.get("reason"), r.get("raw_text","")[:60]) for r in rej])
