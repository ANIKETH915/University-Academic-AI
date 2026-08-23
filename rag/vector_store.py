import os
from typing import Optional, List
import chromadb
from sentence_transformers import SentenceTransformer
from rag.config import CHROMA_DB_DIR, COLLECTION_NAME, EMBEDDING_MODEL_NAME, resolve_collection_name

class VectorStore:
    def __init__(self, persist_directory=CHROMA_DB_DIR, collection_name=None):
        self.persist_directory = persist_directory
        self.collection_name = resolve_collection_name(collection_name)
        
        # Initialize ChromaDB client (Ephemeral for test mode to avoid Windows file lock crashes)
        if os.environ.get("WORKSPACE_DB_TEST_MODE") == "1" or os.environ.get("PYQRAG_TEST_MODE") == "1":
            self.client = chromadb.EphemeralClient()
        else:
            self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        
        # Initialize sentence transformer model
        print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("Embedding model ready.")

    def embed_texts(self, texts):
        return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()

    def add_documents(self, documents, metadatas, ids, batch_size=100):
        if not documents:
            return

        print(f"Adding {len(documents)} documents to vector store...")
        for i in range(0, len(documents), batch_size):
            end_idx = min(i + batch_size, len(documents))
            batch_docs = documents[i:end_idx]
            batch_meta = metadatas[i:end_idx]
            batch_ids = ids[i:end_idx]

            clean_ids = []
            seen_ids = set()
            for idx_val, raw_id in enumerate(batch_ids):
                cid = str(raw_id)
                if cid in seen_ids:
                    cid = f"{cid}_{i + idx_val}"
                seen_ids.add(cid)
                clean_ids.append(cid)

            embeddings = self.embed_texts(batch_docs)
            try:
                self.collection.add(
                    documents=batch_docs,
                    embeddings=embeddings,
                    metadatas=batch_meta,
                    ids=clean_ids,
                )
            except Exception:
                self.collection.upsert(
                    documents=batch_docs,
                    embeddings=embeddings,
                    metadatas=batch_meta,
                    ids=clean_ids,
                )
        print(f"Vector store indexing complete. Total vectors in collection: {self.collection.count()}")

    def replace_documents_for_source(
        self,
        documents,
        metadatas,
        ids,
        *,
        source_file: str,
        workspace_id: Optional[str] = None,
        batch_size: int = 100,
    ):
        """
        Safe replace: insert/upsert NEW vectors first, then delete OLD vectors
        for the same source_file that are not in the new id set.
        If insertion fails, previous vectors remain intact.
        """
        if not documents:
            return 0

        old_ids: List[str] = []
        try:
            if workspace_id:
                res = self.collection.get(
                    where={
                        "$and": [
                            {"source_file": {"$eq": source_file}},
                            {"workspace_id": {"$eq": workspace_id}},
                        ]
                    }
                )
            else:
                res = self.collection.get(where={"source_file": {"$eq": source_file}})
            old_ids = list(res.get("ids") or [])
        except Exception as e:
            raise RuntimeError(
                f"refusing replace for '{source_file}'; cannot list existing vectors: {e}"
            ) from e

        new_ids = [str(i) for i in ids]
        try:
            self.add_documents(documents, metadatas, new_ids, batch_size=batch_size)
        except Exception:
            print(f"[VECTOR_REPLACE] insert failed for '{source_file}'; previous vectors left intact")
            raise

        stale = [oid for oid in old_ids if oid not in set(new_ids)]
        if stale:
            try:
                self.collection.delete(ids=stale)
                print(f"[VECTOR_REPLACE] removed {len(stale)} stale vectors for '{source_file}'")
            except Exception as e:
                print(f"[VECTOR_REPLACE] warn deleting stale: {e}")
        return len(new_ids)

    def search(self, query, doc_type="both", top_k=5, filters=None):
        """
        Hybrid retrieval search combining vector similarity and workspace metadata filtering.
        doc_type: "syllabus", "pyq", or "both"
        """
        query_embedding = self.embed_texts([query])[0]
        
        where_clause = {}
        if filters:
            for k, v in filters.items():
                if v is not None and str(v).strip() != "":
                    where_clause[k] = {"$eq": str(v)}
                    
        if doc_type and doc_type.lower() in ["syllabus", "pyq"]:
            where_clause["doc_type"] = {"$eq": doc_type.lower()}
            
        chroma_where = None
        if len(where_clause) == 1:
            chroma_where = where_clause
        elif len(where_clause) > 1:
            chroma_where = {"$and": [{k: v} for k, v in where_clause.items()]}
            
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=chroma_where,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as ex:
            print(f"VectorStore search error or zero filter matches for {chroma_where}: {ex}")
            return []

        
        formatted_results = []
        if results and results.get("documents") and len(results["documents"]) > 0:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            dists = results["distances"][0]
            
            for doc, meta, dist in zip(docs, metas, dists):
                similarity = max(0.0, 1.0 - (dist / 2.0))
                formatted_results.append({
                    "score": round(similarity, 4),
                    "distance": round(dist, 4),
                    "text": doc,
                    "metadata": meta
                })
                
        return formatted_results

    def delete_by_workspace(self, workspace_id: str):
        """Purges vector entries matching workspace_id from ChromaDB collection."""
        if not workspace_id:
            return 0
        try:
            res = self.collection.get(where={"workspace_id": {"$eq": workspace_id}})
            if res and res.get("ids"):
                ids_to_delete = res["ids"]
                self.collection.delete(ids=ids_to_delete)
                print(f"Purged {len(ids_to_delete)} vectors for workspace_id '{workspace_id}'.")
                return len(ids_to_delete)
        except Exception as e:
            print(f"Error deleting vectors for workspace {workspace_id}: {e}")
        return 0


    def delete_by_source_file(self, filename: str, workspace_id: Optional[str] = None):
        """Purges vector entries for source_file, optionally scoped to workspace_id."""
        if not filename:
            return 0
        try:
            if workspace_id:
                res = self.collection.get(
                    where={
                        "$and": [
                            {"source_file": {"$eq": filename}},
                            {"workspace_id": {"$eq": workspace_id}},
                        ]
                    }
                )
            else:
                res = self.collection.get(where={"source_file": {"$eq": filename}})
            if res and res.get("ids"):
                ids_to_delete = res["ids"]
                self.collection.delete(ids=ids_to_delete)
                scope = f" workspace '{workspace_id}'" if workspace_id else ""
                print(f"Purged {len(ids_to_delete)} vectors for file '{filename}'{scope} from ChromaDB.")
                return len(ids_to_delete)
        except Exception as e:
            print(f"Error deleting vectors for {filename}: {e}")
        return 0

    def get_stats(self):
        count = 0
        syllabus_count = 0
        pyq_count = 0
        try:
            count = self.collection.count()
        except Exception:
            try:
                self.collection = self.client.get_or_create_collection(self.collection_name)
                count = self.collection.count()
            except Exception:
                count = 0

        try:
            if count > 0:
                res = self.collection.get()
                metas = res.get("metadatas", []) if res else []
                for m in metas:
                    dtype = m.get("doc_type")
                    if dtype == "syllabus":
                        syllabus_count += 1
                    elif dtype == "pyq":
                        pyq_count += 1
        except Exception as e:
            print(f"get_stats non-fatal warning: {e}")

        return {
            "total_vectors": count,
            "syllabus_chunks": syllabus_count,
            "pyq_chunks": pyq_count,
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory
        }

