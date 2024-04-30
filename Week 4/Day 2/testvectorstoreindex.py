

import os
import getpass

os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key: ")

##

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

##

from llama_index.readers.wikipedia import WikipediaReader

movie_list = [
    "Dune (2021 film)",
    "Dune: Part Two",
    "The Lord of the Rings: The Fellowship of the Ring",
    "The Lord of the Rings: The Two Towers",
]

wiki_docs = WikipediaReader().load_data(pages=movie_list, auto_suggest=False)
print(len(wiki_docs))

##

from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

client = QdrantClient(location=":memory:")

client.create_collection(
    collection_name="movie_wikis",
    vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
)

##

from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext

vector_store = QdrantVectorStore(client=client, collection_name="movie_wikis")

storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    [],
    storage_context=storage_context,
)

##

from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.extractors import TitleExtractor

pipeline = IngestionPipeline(transformations=[TokenTextSplitter()])

for movie, wiki_doc in zip(movie_list, wiki_docs):
  nodes = pipeline.run(documents=[wiki_doc])
  for node in nodes:
      node.metadata = {"title" : movie}
  index.insert_nodes(nodes)

##

# print('index.vector_store.__dict__ =', index.vector_store.__dict__())
# print('index._vector_store.__dict__ =', index._vector_store.__dict__())
print('index.vector_store.to_dict() =', index.vector_store.to_dict())
print('index._vector_store.to_dict() =', index._vector_store.to_dict())

##

# print(index._data.embedding_dict.keys())

# print(index.docstore.docs.values())

# myretriever = index.as_retriever()
# mydocs = myretriever.retrieve("lord of the rings")
# print(len(mydocs))

# from llama_index.core.extractors import BaseExtractor
# class CustomExtractor(BaseExtractor):
#     def extract(self, nodes):
#         metadata_list = [node.metadata for node in nodes]
#         return metadata_list
# mytransformations = [TokenTextSplitter()] + [CustomExtractor()]
# mypipeline = IngestionPipeline(transformations=mytransformations)
# mypipeline.run()
