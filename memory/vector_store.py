"""
Vector Store

ChromaDB-based vector store for semantic search and episodic memory.
Stores past analyses, patterns, and signal contexts.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import hashlib
import json

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class VectorStore:
    """
    Vector store using ChromaDB for semantic memory.

    Stores and retrieves:
    - Past signal analyses
    - Trading patterns
    - Market contexts
    """

    def __init__(
        self,
        persist_directory: str = "data/chromadb",
        collection_name: str = "cryptoalpha",
    ):
        """
        Initialize the vector store.

        Args:
            persist_directory: Directory to persist the database
            collection_name: Name of the collection
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        if CHROMADB_AVAILABLE:
            # Create persistent client
            os.makedirs(persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False),
            )

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "CryptoAlpha episodic memory"},
            )
        else:
            self.client = None
            self.collection = None
            self._fallback_store: List[Dict] = []

    def _generate_id(self, text: str, metadata: Dict) -> str:
        """Generate a unique ID for a document."""
        content = text + json.dumps(metadata, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()

    def add_analysis(
        self,
        symbol: str,
        analysis_type: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add an analysis to the vector store.

        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            analysis_type: Type of analysis ('technical', 'onchain', 'signal')
            content: Text content of the analysis
            metadata: Additional metadata

        Returns:
            Document ID
        """
        doc_metadata = {
            "symbol": symbol,
            "analysis_type": analysis_type,
            "timestamp": datetime.now().isoformat(),
            **(metadata or {}),
        }

        doc_id = self._generate_id(content, doc_metadata)

        if CHROMADB_AVAILABLE and self.collection:
            self.collection.add(
                documents=[content],
                metadatas=[doc_metadata],
                ids=[doc_id],
            )
        else:
            self._fallback_store.append({
                "id": doc_id,
                "document": content,
                "metadata": doc_metadata,
            })

        return doc_id

    def add_signal(
        self,
        symbol: str,
        signal_type: str,
        confidence: float,
        reasoning: str,
        outcome: Optional[str] = None,
        pnl_pct: Optional[float] = None,
    ) -> str:
        """
        Add a signal to memory.

        Args:
            symbol: Trading symbol
            signal_type: BUY/SELL/HOLD
            confidence: Signal confidence
            reasoning: Full reasoning text
            outcome: Win/Loss/Pending
            pnl_pct: Profit/loss percentage

        Returns:
            Document ID
        """
        metadata = {
            "signal_type": signal_type,
            "confidence": confidence,
            "outcome": outcome or "pending",
            "pnl_pct": pnl_pct or 0,
        }

        return self.add_analysis(
            symbol=symbol,
            analysis_type="signal",
            content=reasoning,
            metadata=metadata,
        )

    def add_pattern(
        self,
        pattern_type: str,
        description: str,
        symbols: List[str],
        success_rate: float,
        sample_size: int,
    ) -> str:
        """
        Add a trading pattern to memory.

        Args:
            pattern_type: Type of pattern
            description: Pattern description
            symbols: Symbols where pattern occurred
            success_rate: Historical success rate
            sample_size: Number of occurrences

        Returns:
            Document ID
        """
        metadata = {
            "pattern_type": pattern_type,
            "symbols": ",".join(symbols),
            "success_rate": success_rate,
            "sample_size": sample_size,
        }

        return self.add_analysis(
            symbol="PATTERN",
            analysis_type="pattern",
            content=description,
            metadata=metadata,
        )

    def search(
        self,
        query: str,
        n_results: int = 5,
        symbol: Optional[str] = None,
        analysis_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.

        Args:
            query: Search query
            n_results: Number of results to return
            symbol: Filter by symbol
            analysis_type: Filter by analysis type

        Returns:
            List of matching documents with metadata
        """
        where_filter = {}
        if symbol:
            where_filter["symbol"] = symbol
        if analysis_type:
            where_filter["analysis_type"] = analysis_type

        if CHROMADB_AVAILABLE and self.collection:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_filter if where_filter else None,
            )

            documents = []
            for i, doc_id in enumerate(results["ids"][0]):
                documents.append({
                    "id": doc_id,
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if results.get("distances") else None,
                })
            return documents
        else:
            # Fallback: simple text matching
            matches = []
            query_lower = query.lower()

            for doc in self._fallback_store:
                if query_lower in doc["document"].lower():
                    if symbol and doc["metadata"].get("symbol") != symbol:
                        continue
                    if analysis_type and doc["metadata"].get("analysis_type") != analysis_type:
                        continue
                    matches.append(doc)

            return matches[:n_results]

    def get_similar_signals(
        self,
        symbol: str,
        current_analysis: str,
        n_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Find similar past signals for a symbol.

        Args:
            symbol: Trading symbol
            current_analysis: Current analysis text
            n_results: Number of results

        Returns:
            List of similar past signals
        """
        return self.search(
            query=current_analysis,
            n_results=n_results,
            symbol=symbol,
            analysis_type="signal",
        )

    def get_winning_patterns(
        self,
        min_success_rate: float = 0.6,
        min_sample_size: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Get patterns with high success rates.

        Args:
            min_success_rate: Minimum success rate
            min_sample_size: Minimum sample size

        Returns:
            List of winning patterns
        """
        if CHROMADB_AVAILABLE and self.collection:
            # Get all patterns
            results = self.collection.get(
                where={"analysis_type": "pattern"},
            )

            patterns = []
            for i, doc_id in enumerate(results["ids"]):
                metadata = results["metadatas"][i]
                if (metadata.get("success_rate", 0) >= min_success_rate and
                    metadata.get("sample_size", 0) >= min_sample_size):
                    patterns.append({
                        "id": doc_id,
                        "document": results["documents"][i],
                        "metadata": metadata,
                    })

            return sorted(patterns, key=lambda x: -x["metadata"]["success_rate"])
        else:
            return [
                doc for doc in self._fallback_store
                if (doc["metadata"].get("analysis_type") == "pattern" and
                    doc["metadata"].get("success_rate", 0) >= min_success_rate and
                    doc["metadata"].get("sample_size", 0) >= min_sample_size)
            ]

    def get_symbol_history(
        self,
        symbol: str,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Get historical analyses for a symbol.

        Args:
            symbol: Trading symbol
            limit: Maximum results

        Returns:
            List of historical analyses
        """
        if CHROMADB_AVAILABLE and self.collection:
            results = self.collection.get(
                where={"symbol": symbol},
                limit=limit,
            )

            documents = []
            for i, doc_id in enumerate(results["ids"]):
                documents.append({
                    "id": doc_id,
                    "document": results["documents"][i],
                    "metadata": results["metadatas"][i],
                })

            # Sort by timestamp
            documents.sort(
                key=lambda x: x["metadata"].get("timestamp", ""),
                reverse=True,
            )
            return documents
        else:
            matches = [
                doc for doc in self._fallback_store
                if doc["metadata"].get("symbol") == symbol
            ]
            return sorted(
                matches,
                key=lambda x: x["metadata"].get("timestamp", ""),
                reverse=True,
            )[:limit]

    def update_signal_outcome(
        self,
        doc_id: str,
        outcome: str,
        pnl_pct: float,
    ) -> bool:
        """
        Update a signal's outcome.

        Args:
            doc_id: Document ID
            outcome: Win/Loss
            pnl_pct: Profit/loss percentage

        Returns:
            Success status
        """
        if CHROMADB_AVAILABLE and self.collection:
            try:
                # Get current document
                result = self.collection.get(ids=[doc_id])
                if not result["ids"]:
                    return False

                # Update metadata
                metadata = result["metadatas"][0]
                metadata["outcome"] = outcome
                metadata["pnl_pct"] = pnl_pct
                metadata["updated_at"] = datetime.now().isoformat()

                self.collection.update(
                    ids=[doc_id],
                    metadatas=[metadata],
                )
                return True
            except Exception:
                return False
        else:
            for doc in self._fallback_store:
                if doc["id"] == doc_id:
                    doc["metadata"]["outcome"] = outcome
                    doc["metadata"]["pnl_pct"] = pnl_pct
                    return True
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        if CHROMADB_AVAILABLE and self.collection:
            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory,
                "backend": "chromadb",
            }
        else:
            return {
                "total_documents": len(self._fallback_store),
                "collection_name": self.collection_name,
                "backend": "fallback",
            }

    def clear(self) -> None:
        """Clear all documents."""
        if CHROMADB_AVAILABLE and self.collection:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
            )
        else:
            self._fallback_store = []


def create_vector_store(
    persist_directory: str = "data/chromadb",
) -> VectorStore:
    """Create a VectorStore instance."""
    return VectorStore(persist_directory=persist_directory)


if __name__ == "__main__":
    print("Testing VectorStore...")

    store = VectorStore(persist_directory="data/test_chromadb")

    # Test adding analyses
    print("\n1. Adding analyses...")

    doc_id1 = store.add_analysis(
        symbol="BTC/USDT",
        analysis_type="technical",
        content="RSI is oversold at 28, price testing 200 EMA support at $42,000. "
                "MACD showing bullish divergence. Volume increasing on down moves.",
    )
    print(f"  Added technical analysis: {doc_id1[:8]}...")

    doc_id2 = store.add_signal(
        symbol="BTC/USDT",
        signal_type="BUY",
        confidence=0.72,
        reasoning="Multiple bullish signals converging: RSI oversold, whale accumulation "
                  "detected (1,500 BTC in 24h), price at major support. Pattern similar to "
                  "November 2023 reversal setup.",
    )
    print(f"  Added signal: {doc_id2[:8]}...")

    doc_id3 = store.add_pattern(
        pattern_type="whale_accumulation_reversal",
        description="When whales accumulate >1000 BTC while RSI is below 30 and price "
                    "is at 200 EMA, historical probability of 15%+ move within 7 days is 68%.",
        symbols=["BTC/USDT", "ETH/USDT"],
        success_rate=0.68,
        sample_size=25,
    )
    print(f"  Added pattern: {doc_id3[:8]}...")

    # Test searching
    print("\n2. Searching...")

    results = store.search("RSI oversold support", n_results=3)
    print(f"  Found {len(results)} results for 'RSI oversold support'")

    # Test similar signals
    print("\n3. Finding similar signals...")

    similar = store.get_similar_signals(
        symbol="BTC/USDT",
        current_analysis="RSI at 25, price near support, whales buying",
        n_results=3,
    )
    print(f"  Found {len(similar)} similar past signals")

    # Test winning patterns
    print("\n4. Getting winning patterns...")

    patterns = store.get_winning_patterns(min_success_rate=0.6)
    print(f"  Found {len(patterns)} winning patterns")

    # Test stats
    print("\n5. Store stats...")
    stats = store.get_stats()
    print(f"  Total documents: {stats['total_documents']}")
    print(f"  Backend: {stats['backend']}")

    # Cleanup
    store.clear()
    print("\nTest complete!")
