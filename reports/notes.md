1. llamaparse cannot parse tables, algotithms, mathematical fromulas well.

2. FAISS index does not work well on specific keywords, so added bm25 index to support keyword search. used rrf to combine results from both indices. configurate bm25 to preserve technical compounds like ML-KEM.KeyGen

3. for search, added query variants to support different ways of asking the same question. for example, "key generation" can be asked as "ML-KEM.KeyGen key generation". this is done by looking for specific keywords in the query and adding variants accordingly.

4. should add a threshold for score of retrieved documents, if all documents are below threshold, do query expansion to try to retrieve more relevant documents.

5. can let llm do a query evaluation step to decide whether we need to do query expansion or not. 

6. unit tests to test whether retry retrieval with query expansion works as expected.

# state of art RAG system design
## technical document characteristics
- Dense Technical Jargon & Acronyms: Expect heavy use of highly specific terminology (e.g., AES-256-GCM, FIPS 140-3, NIST SP 800-53, FedRAMP, PKI, Zero Trust). Semantic meaning is heavily tied to these exact phrases.

- Complex Formatting: These documents (often PDFs and Word docs) rely heavily on tables (e.g., compliance checklists, port/protocol matrices), architectural diagrams, and **hierarchical** headers to convey critical information.

- Extreme Length and Cross-Referencing: Enterprise RFPs (Requests for Proposals) and government security standards can span hundreds of pages and constantly reference other internal sections or external regulations.

- Strict Versioning and Obsolescence: Cybersecurity is highly **time-sensitive**. A solution document from 2022 might recommend TLS 1.2, while a 2026 document mandates TLS 1.3. Using outdated context is a critical failure.

- High Density of Constraints: Statements often contain rigorous conditions ("Component X must be used unless deployed in a disconnected environment, in which case Component Y is required").
 
## indexing
### ingestion

### chunking
1. recursive
2. 

### embedding

### index construction
1. multi representation: create multiple indices with different representations of the same document (e.g., one with tables flattened, one with tables preserved, one with hierarchical headers). this allows for more flexible retrieval based on the query.

2. RAPTOR: 

3. ColBERT

## query translation/expansion
1. multi-query: rewrite the query into multiple variants to capture different ways of asking the same question. for example, "key generation" can be asked as "ML-KEM.KeyGen key generation". this is done by looking for specific keywords in the query and adding variants accordingly.

2. Rag fusion

3. break into subqueries 

4. **step-back prompting**: go more abstract to capture the underlying intent of the query, rather than specific keywords. for example, if the query is "what are the requirements for key generation?", the step-back prompt might be "what are the requirements for cryptographic operations?" this can help capture relevant information that may not use the exact keywords in the original query.

5. **HyDE**: generate a hypothetical answer to the query using the LLM, then use that answer to retrieve relevant documents. this can help capture relevant information that may not be directly related to the original query, but is still relevant to the underlying intent.

## routing

1. logical routing: based on the query, determine which index to use (e.g., if the query contains specific keywords, use the bm25 index, otherwise use the faiss index).
2. semantic rounting:

## query construction
structured output