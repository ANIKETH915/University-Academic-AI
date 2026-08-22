import os
import unittest
import fitz
from rag.workspace_db import WorkspaceDB
from rag.vector_store import VectorStore
from rag.dynamic_ingest import DynamicIngestPipeline
from rag.pyq_intelligence import PYQIntelligenceEngine
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
    "Transformer Self Attention Multi Head Scaled Dot Product",
    "Batch Normalization Layer Normalization Covariate Shift",
    "Hyperparameter Tuning Learning Rate Schedule Grid Search",
    "Loss Functions Cross Entropy Mean Squared Margin Loss",
    "Object Detection YOLO Faster RCNN Feature Pyramid",
    "Semantic Segmentation UNET DeepLab Mask RCNN Architecture",
    "Reinforcement Learning Q Learning Policy Gradient Actor Critic",
    "Natural Language Processing Word Embeddings Word2Vec GloVe",
    "Dimensionality Reduction PCA tSNE UMAP Feature Extraction",
    "Model Evaluation Precision Recall F1 Score ROC AUC Curve",
    "Graph Neural Networks Node Classification Edge Prediction",
    "Attention Mechanism Query Key Value Vector Operations",
    "Positional Encoding Sinusoidal Temporal Sequence Modeling",
    "Vision Transformers Patch Embeddings Class Token Linear Projection",
    "Contrastive Learning SimCLR MoCo Self Supervised Representation"
]

def create_synthetic_pdf(num_questions: int, filename: str, doc_title: str = "COMPUTE_ENG_EXAM") -> str:
    """Helper to generate a clean synthetic PDF with exact number of questions."""
    doc = fitz.open()
    lines = [
        f"UNIVERSITY EXAMINATION — {doc_title}",
        "SUBJECT: ADVANCED ALGORITHMS | SEMESTER 7",
        "Duration: 3 Hours | Max Marks: 80",
        "Instructions: Attempt all questions.",
        ""
    ]

    for q_i in range(1, num_questions + 1):
        topic = TOPICS_POOL[(q_i - 1) % len(TOPICS_POOL)]
        # Keep lines short so PyMuPDF insert_text does not truncate mid-question
        short_topic = " ".join(topic.split()[:6])
        if num_questions <= 5:
            q_str = f"Q1({chr(96 + q_i)}) Explain {short_topic} in detail with suitable diagram. [5 Marks]"
        else:
            p_num = (q_i - 1) // 5 + 1
            s_char = chr(97 + ((q_i - 1) % 5))
            q_str = f"Q{p_num}({s_char}) Describe {short_topic} with examples and diagrams. [10 Marks]"
        lines.append(q_str)

    # Paginate to avoid clipping
    y = 50
    page = doc.new_page()
    for line in lines:
        if y > 750:
            page = doc.new_page()
            y = 50
        page.insert_text((50, y), line, fontsize=9)
        y += 14

    pdf_path = os.path.join("scratch", filename)
    os.makedirs("scratch", exist_ok=True)
    doc.save(pdf_path)
    doc.close()
    return pdf_path


class TestVariableQuestionCountsSuite(unittest.TestCase):
    """
    Test suite verifying that PYQ ingestion & analytics scale dynamically
    for ANY number of uploaded PDFs and ANY question count per paper.
    """

    def setUp(self):
        self.ws_db = WorkspaceDB()
        self.store = VectorStore()
        self.pipeline = DynamicIngestPipeline()
        self.engine = PYQIntelligenceEngine()

    def tearDown(self):
        # Clean up scratch pdfs created during test
        if os.path.exists("scratch"):
            for f in os.listdir("scratch"):
                if f.startswith("test_synth_"):
                    try:
                        os.remove(os.path.join("scratch", f))
                    except Exception:
                        pass

    def test_01_single_pdf_5_questions(self):
        ws_id = "ws-var-test-5q"
        self.ws_db.delete_workspace(ws_id)
        ws = self.ws_db.get_or_create(ws_id, subject="Algorithms")
        self.store.delete_by_workspace(ws_id)

        fpath = create_synthetic_pdf(5, "test_synth_5q.pdf")
        metas = self.pipeline.parse_pyq_pdf(fpath, ws)
        self.assertEqual(len(metas), 5, "Expected exactly 5 questions extracted for 5-question PDF")

        analysis = self.engine.get_pyq_analysis(ws_id)
        self.assertEqual(analysis["total_papers"], 1)
        self.assertEqual(analysis["total_questions_analyzed"], 5)

        self.ws_db.delete_workspace(ws_id)
        self.store.delete_by_workspace(ws_id)

    def test_02_single_pdf_10_questions(self):
        ws_id = "ws-var-test-10q"
        self.ws_db.delete_workspace(ws_id)
        ws = self.ws_db.get_or_create(ws_id, subject="Algorithms")
        self.store.delete_by_workspace(ws_id)

        fpath = create_synthetic_pdf(10, "test_synth_10q.pdf")
        metas = self.pipeline.parse_pyq_pdf(fpath, ws)
        self.assertEqual(len(metas), 10, "Expected exactly 10 questions extracted for 10-question PDF")

        analysis = self.engine.get_pyq_analysis(ws_id)
        self.assertEqual(analysis["total_papers"], 1)
        self.assertEqual(analysis["total_questions_analyzed"], 10)

        self.ws_db.delete_workspace(ws_id)
        self.store.delete_by_workspace(ws_id)

    def test_03_single_pdf_15_questions(self):
        ws_id = "ws-var-test-15q"
        self.ws_db.delete_workspace(ws_id)
        ws = self.ws_db.get_or_create(ws_id, subject="Algorithms")
        self.store.delete_by_workspace(ws_id)

        fpath = create_synthetic_pdf(15, "test_synth_15q.pdf")
        metas = self.pipeline.parse_pyq_pdf(fpath, ws)
        self.assertEqual(len(metas), 15, "Expected exactly 15 questions extracted for 15-question PDF")

        analysis = self.engine.get_pyq_analysis(ws_id)
        self.assertEqual(analysis["total_papers"], 1)
        self.assertEqual(analysis["total_questions_analyzed"], 15)

        self.ws_db.delete_workspace(ws_id)
        self.store.delete_by_workspace(ws_id)

    def test_04_single_pdf_20_questions(self):
        ws_id = "ws-var-test-20q"
        self.ws_db.delete_workspace(ws_id)
        ws = self.ws_db.get_or_create(ws_id, subject="Algorithms")
        self.store.delete_by_workspace(ws_id)

        fpath = create_synthetic_pdf(20, "test_synth_20q.pdf")
        metas = self.pipeline.parse_pyq_pdf(fpath, ws)
        self.assertEqual(len(metas), 20, "Expected exactly 20 questions extracted for 20-question PDF")

        analysis = self.engine.get_pyq_analysis(ws_id)
        self.assertEqual(analysis["total_papers"], 1)
        self.assertEqual(analysis["total_questions_analyzed"], 20)

        self.ws_db.delete_workspace(ws_id)
        self.store.delete_by_workspace(ws_id)

    def test_05_multi_pdf_variable_sum(self):
        """4 PDFs with 5 + 10 + 15 + 20 questions -> 50 Total Questions across 4 Papers"""
        ws_id = "ws-var-test-multi-50"
        self.ws_db.delete_workspace(ws_id)
        ws = self.ws_db.get_or_create(ws_id, subject="Algorithms")
        self.store.delete_by_workspace(ws_id)

        counts = [5, 10, 15, 20]
        total_expected = sum(counts)

        for idx, count in enumerate(counts):
            fpath = create_synthetic_pdf(count, f"test_synth_paper_{idx}_{count}q.pdf")
            self.pipeline.parse_pyq_pdf(fpath, ws)

        analysis = self.engine.get_pyq_analysis(ws_id)
        self.assertEqual(analysis["total_papers"], 4, "Expected 4 analyzed papers")
        self.assertEqual(analysis["total_questions_analyzed"], total_expected, f"Expected exactly {total_expected} questions analyzed")

        self.ws_db.delete_workspace(ws_id)
        self.store.delete_by_workspace(ws_id)

    def test_06_ten_pdfs_scale_test(self):
        """10 PDFs with different question counts -> Sum of actual counts"""
        ws_id = "ws-var-test-10-pdfs"
        self.ws_db.delete_workspace(ws_id)
        ws = self.ws_db.get_or_create(ws_id, subject="Computer Architecture")
        self.store.delete_by_workspace(ws_id)

        counts = [5, 8, 10, 12, 15, 18, 20, 7, 9, 14]
        total_expected = sum(counts)

        for idx, count in enumerate(counts):
            fpath = create_synthetic_pdf(count, f"test_synth_scale_{idx}_{count}q.pdf")
            self.pipeline.parse_pyq_pdf(fpath, ws)

        analysis = self.engine.get_pyq_analysis(ws_id)
        self.assertEqual(analysis["total_papers"], 10, "Expected 10 papers analyzed")
        self.assertEqual(analysis["total_questions_analyzed"], total_expected, f"Expected exactly {total_expected} total questions analyzed")

        self.ws_db.delete_workspace(ws_id)
        self.store.delete_by_workspace(ws_id)

    def test_07_one_document_failure_isolation(self):
        """If one document fails quality validation, valid documents remain intact and READY."""
        ws_id = "ws-var-test-fail-isolation"
        self.ws_db.delete_workspace(ws_id)
        ws = self.ws_db.get_or_create(ws_id, subject="Systems")
        self.store.delete_by_workspace(ws_id)

        fpath1 = create_synthetic_pdf(10, "test_synth_valid1.pdf")
        self.pipeline.parse_pyq_pdf(fpath1, ws)

        fpath_bad = os.path.join("scratch", "test_synth_corrupt.pdf")
        with open(fpath_bad, "w", encoding="utf-8") as f:
            f.write("GARBAGE NOT A PDF DATA")

        try:
            self.pipeline.parse_pyq_pdf(fpath_bad, ws)
        except Exception:
            pass

        analysis = self.engine.get_pyq_analysis(ws_id)
        self.assertEqual(analysis["total_papers"], 1, "Expected 1 valid paper remaining READY in workspace analytics")
        self.assertEqual(analysis["total_questions_analyzed"], 10, "Expected 10 questions from valid document")

        self.ws_db.delete_workspace(ws_id)
        self.store.delete_by_workspace(ws_id)

    def test_08_real_deep_learning_pdfs_quality_gate(self):
        """
        Real MU Deep Learning PDFs: readable text must extract canonical questions;
        font-encoded/garbled pages must not invent garbage intelligence.
        """
        dl_dir = "d:/pyqrag/data/pyq/deep-learning"
        if not os.path.isdir(dl_dir):
            self.skipTest(f"Real DL PDF fixtures not present at {dl_dir}")

        ws_id = "ws-var-test-dl-regression"
        self.ws_db.delete_workspace(ws_id)
        ws = self.ws_db.get_or_create(ws_id, subject="Deep Learning")
        self.store.delete_by_workspace(ws_id)

        dl_files = [
            os.path.join(dl_dir, f) for f in os.listdir(dl_dir)
            if f.endswith(".pdf") and "Copy" not in f and "compressed" not in f
        ]
        if not dl_files:
            self.skipTest("No Deep Learning PDF fixtures found")

        per_file = {}
        for fpath in dl_files:
            metas = self.pipeline.parse_pyq_pdf(fpath, ws)
            per_file[os.path.basename(fpath)] = len(metas)
            # No numeric garbage IDs
            for m in metas:
                qid = str(m.get("question_id", ""))
                self.assertRegex(qid, r"^Q\d+(\([a-zivx]+\))?$", f"Invalid ID {qid}")

        total_extracted = sum(per_file.values())
        # At least one readable paper should yield real questions
        self.assertGreaterEqual(total_extracted, 10, f"Expected readable PYQ extraction, got {per_file}")
        analysis = self.engine.get_pyq_analysis(ws_id)
        self.assertEqual(analysis["total_questions_analyzed"], total_extracted)
        # Main intelligence payload must not dump source questions
        self.assertEqual(analysis.get("extracted_questions"), [])

        self.ws_db.delete_workspace(ws_id)
        self.store.delete_by_workspace(ws_id)


if __name__ == "__main__":
    unittest.main()
