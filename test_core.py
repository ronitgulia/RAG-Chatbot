import unittest
from rag_pipeline import ConversationMemory
from vector_store import VectorStoreManager

class TestConversationMemory(unittest.TestCase):
    def test_sliding_window(self):
        mem = ConversationMemory(max_turns=2)
        mem.add("user", "Hello")
        mem.add("assistant", "Hi there")
        mem.add("user", "How are you?")
        mem.add("assistant", "I'm good")
        self.assertEqual(len(mem), 4)
        
        # Add one more pair to trigger sliding window
        mem.add("user", "What's up?")
        mem.add("assistant", "Nothing much")
        self.assertEqual(len(mem), 4)
        
        history = mem.get_history()
        self.assertEqual(history[0]["content"], "How are you?")
        self.assertEqual(history[-1]["content"], "Nothing much")

    def test_format_history(self):
        mem = ConversationMemory()
        self.assertEqual(mem.format_history(), "No previous conversation.")
        mem.add("user", "Hello")
        self.assertIn("User: Hello", mem.format_history())

    def test_clear(self):
        mem = ConversationMemory()
        mem.add("user", "Test")
        mem.clear()
        self.assertEqual(len(mem), 0)

class TestVectorStoreManager(unittest.TestCase):
    def test_hybrid_search_rrf_logic(self):
        # Pass a mock embedding model to avoid loading heavy sentence-transformers
        class MockEmbeddingModel:
            def get_langchain_embeddings(self):
                return None
        
        vsm = VectorStoreManager(embedding_model=MockEmbeddingModel())
        
        dense_results = [
            {"page_content": "Doc A", "metadata": {"source": "A"}, "score": 0.9},
            {"page_content": "Doc B", "metadata": {"source": "B"}, "score": 0.8},
        ]
        
        sparse_results = [
            {"page_content": "Doc B", "metadata": {"source": "B"}, "score": 2.5},
            {"page_content": "Doc C", "metadata": {"source": "C"}, "score": 1.5},
        ]
        
        # Override retrieval methods to isolate hybrid logic
        vsm.dense_search = lambda q, top_k: dense_results
        vsm.sparse_search = lambda q, top_k: sparse_results
        
        # Hybrid search with alpha=0.5 (equal weighting)
        results = vsm.hybrid_search("dummy query", top_k=3, alpha=0.5)
        
        self.assertEqual(len(results), 3)
        # Doc B appears in both (rank 1 in dense, rank 0 in sparse)
        # Doc A appears in dense (rank 0)
        # Doc C appears in sparse (rank 1)
        # RRF formula gives highest score to Doc B because it appears in both lists.
        self.assertEqual(results[0]["page_content"], "Doc B")
        self.assertTrue("rrf_score" in results[0])
        self.assertEqual(results[0]["search_type"], "hybrid")
        self.assertEqual(results[1]["page_content"], "Doc A")
        self.assertEqual(results[2]["page_content"], "Doc C")

if __name__ == "__main__":
    unittest.main()
