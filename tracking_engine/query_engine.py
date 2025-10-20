"""
Optimized RAG pipeline with 4-bit quantization and proper answer extraction.
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from template.embeddings_store import embeddings_store
from operations.conversation_memory import conversation_manager
from template.config import settings
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class QueryEngine:
    """
    RAG pipeline: Retrieve → Rerank → Generate
    """
    
    def __init__(self):
        self.reranker_tokenizer = None
        self.reranker_model = None
        self.llm_tokenizer = None
        self.llm_model = None
        
        self._init_models()
    
    def _init_models(self):
        """Initialize models with 4-bit quantization"""
        
        # --- Reranker (smaller, no quantization needed) ---
        logger.info(f"Loading reranker: {settings.RERANKER_MODEL}")
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(
            settings.RERANKER_MODEL
        )
        self.reranker_model = AutoModelForSequenceClassification.from_pretrained(
            settings.RERANKER_MODEL
        )
        self.reranker_model.eval()
        logger.info("Reranker loaded")
        
        # --- LLM with 4-bit quantization ---
        logger.info(f"Loading LLM: {settings.LLAMA_MODEL}")
        
        if settings.USE_4BIT_QUANT:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=settings.BNB_4BIT_QUANT_TYPE,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            logger.info("Using 4-bit quantization (saves ~75% VRAM)")
        else:
            bnb_config = None
        
        self.llm_tokenizer = AutoTokenizer.from_pretrained(settings.LLAMA_MODEL)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            settings.LLAMA_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16 if not settings.USE_4BIT_QUANT else None
        )
        self.llm_model.eval()
        
        logger.info(f"LLM loaded on {self.llm_model.device}")
    
    # def retrieve_and_answer(
    #     self, 
    #     query: str,
    #     conversation_id: Optional[str] = None,
    #     include_history: bool = True
    # ) -> Dict:
    #     """
    #     Main RAG pipeline.
    #     """
        
    #     # --- Step 1: Get conversation context ---
    #     conversation = None
    #     context_history = ""
        
    #     if conversation_id and include_history:
    #         conversation = conversation_manager.get_conversation(conversation_id)
    #         context_history = conversation.get_context_string(max_turns=3)
        
    #     # --- Step 2: Retrieve candidates ---
    #     logger.info(f"🔍 Retrieving top {settings.TOP_K} candidates...")
    #     hits = embeddings_store.search(query, top_k=settings.TOP_K)
        
    #     if not hits:
    #         return {
    #             "answer": "I couldn't find relevant information in the factsheets. Could you rephrase your question?",
    #             "citations": [],
    #             "cached": False
    #         }
        
    #     # --- Step 3: Rerank ---
    #     logger.info(f"🎯 Reranking to top {settings.RERANK_TOP_N}...")
    #     reranked = self._rerank_documents(query, hits)
        
    #     # --- Step 4: Build context ---
    #     context = self._build_context(reranked)
        
    #     # --- Step 5: Generate answer ---
    #     logger.info("🤖 Generating response...")
    #     answer = self._generate_answer(query, context, context_history)
        
    #     # --- Step 6: Save to conversation ---
    #     if conversation:
    #         retrieved_texts = [r["text"][:100] for r in reranked]
    #         conversation.add_turn(query, answer, retrieved_texts)
        
    #     return {
    #         "answer": answer,
    #         "citations": [r["metadata"] for r in reranked],
    #         "num_sources": len(reranked),
    #         "cached": False
    #     }
    def retrieve_and_answer(
        self, 
        query: str,
        conversation_id: Optional[str] = None,
        include_history: bool = True
        ) -> Dict:
        """Simple, working RAG pipeline"""

        # Step 1: Conversation context
        conversation = None
        context_history = ""
        if conversation_id and include_history:
            conversation = conversation_manager.get_conversation(conversation_id)
            context_history = conversation.get_context_string(max_turns=3)

        # Step 2: Simple search (WORKS!)
        logger.info(f"🔍 Searching for: {query}")
        hits = embeddings_store.search(query, top_k=settings.TOP_K)

        if not hits:
            return {
                "answer": "I couldn't find relevant information in the factsheets.",
                "citations": [],
                "num_sources": 0,
                "cached": False
            }

        # Step 3: Rerank
        reranked = self._rerank_documents(query, hits)

        # Step 4: Build context
        context = self._build_context(reranked)

        # Step 5: Generate answer
        answer = self._generate_answer(query, context, context_history)

        # Step 6: Save conversation
        if conversation:
            retrieved_texts = [r["text"][:100] for r in reranked]
            conversation.add_turn(query, answer, retrieved_texts)

        return {
            "answer": answer,
            "citations": [r["metadata"] for r in reranked],
            "num_sources": len(reranked),
            "cached": False
        }

    
    def _rerank_documents(
        self, 
        query: str, 
        hits: List[Dict]
    ) -> List[Dict]:
        """Rerank using cross-encoder"""
        
        if len(hits) <= settings.RERANK_TOP_N:
            return hits
        
        try:
            # Create query-document pairs
            pairs = [[query, hit["text"]] for hit in hits]
            
            # Batch encode
            inputs = self.reranker_tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                scores = self.reranker_model(**inputs).logits.squeeze(-1)
            
            # Sort by reranker scores
            scores_list = scores.cpu().numpy().tolist()
            ranked = sorted(
                zip(hits, scores_list),
                key=lambda x: x[1],
                reverse=True
            )[:settings.RERANK_TOP_N]
            
            return [r[0] for r in ranked]
        
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return hits[:settings.RERANK_TOP_N]
    
    @staticmethod
    def _build_context(documents: List[Dict]) -> str:
        """Build context string from documents"""
        context_parts = []
        
        for idx, doc in enumerate(documents, 1):
            text = doc["text"]
            meta = doc["metadata"]
            
            source_info = f"[Source {idx}: Page {meta.get('page_number', 'N/A')}]"
            context_parts.append(f"{source_info}\n{text}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def _generate_answer(
        self, 
        query: str, 
        context: str,
        conversation_history: str = ""
    ) -> str:
        """Generate answer using LLM"""
        
        # Build prompt with Llama 3.1 format
        system_prompt = (
            "You are a helpful financial assistant analyzing AMC mutual fund factsheets. "
            "Answer questions accurately based on the provided context. "
            "If the answer isn't in the context, say so clearly. "
            "Be concise and specific. Cite page numbers when relevant."
        )
        
        user_content = f"""Context from factsheets:
    {context}

    Question: {query}

    Provide a clear, accurate answer based solely on the context above."""
        
        if conversation_history:
            user_content = f"Previous conversation:\n{conversation_history}\n\n{user_content}"
        
        # Format for Llama 3.1 Instruct
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        # Apply chat template
        prompt = self.llm_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Generate
        inputs = self.llm_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.llm_model.device)
        
        with torch.no_grad():
            outputs = self.llm_model.generate(
                **inputs,
                max_new_tokens=settings.MAX_NEW_TOKENS,
                temperature=settings.TEMPERATURE,
                top_p=settings.TOP_P,
                do_sample=True,
                pad_token_id=self.llm_tokenizer.eos_token_id
            )
        
        # Decode ONLY the generated tokens (not the prompt)
        # This is the key fix!
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]  # Skip prompt tokens
        
        answer = self.llm_tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        )
        
        # Clean up any remaining artifacts
        answer = answer.strip()
        for token in ["<|eot_id|>", "<|end_of_text|>", "</s>"]:
            answer = answer.replace(token, "")
        
        return answer.strip()
    
    @staticmethod
    def _extract_answer(full_response: str, prompt: str) -> str:
        """Extract only the generated answer, removing the prompt"""
        
        # Remove the input prompt
        if prompt in full_response:
            answer = full_response.split(prompt)[-1]
        else:
            answer = full_response
        
        # Clean up common artifacts
        answer = answer.strip()
        
        # Remove any trailing special tokens
        for token in ["<|eot_id|>", "<|end_of_text|>", "</s>"]:
            answer = answer.replace(token, "")
        
        return answer.strip()


# Singleton instance
query_engine = QueryEngine()