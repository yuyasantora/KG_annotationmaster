import faiss
import os
import time
import numpy as np
from typing import List, Tuple
import pickle

def create_index_faiss(folders: List[str], index_path: str='index.faiss') -> faiss.Index:
    start = time.time()
    dimension = 1536 # You should specify the dimension of your vectors
    # If index doesn't exist, create it
    index = faiss.IndexFlatL2(dimension)
    file_vector_pairs = []
    for folder in folders:
        for file in os.listdir(folder):
            if file.endswith('.npy'):
                a = np.load(os.path.join(folder, file))
                vec = a.astype('float32')  # 必要に応じて形状を調整
                file = file.replace(".npy", "")
                file_vector_pairs.append((folder + "/" + file, vec))
    # Add all vectors to the index
    vectors = np.vstack([pair[1] for pair in file_vector_pairs])
    
    index.add(vectors)
    
    # Save the index
    faiss.write_index(index, index_path)
    end = time.time()
    print("Faiss Index created in", end - start, "seconds")
    # Save the file_vector_pairs to a pickle file
    with open('file_vector_pairs.pkl', 'wb') as f:
        pickle.dump(file_vector_pairs, f)
    return index

def find_closest_vectors_faiss(ref: np.ndarray, index_path: str, top_k: int) -> List[Tuple[str, float]]:
    with open('file_vector_pairs.pkl', 'rb') as f:
        file_vector_pairs = pickle.load(f)
    # Search the top_k closest vectors
    start = time.time()
    index = faiss.read_index(index_path)
    _, indices = index.search(ref[np.newaxis, :].astype('float32'), top_k)
    # Sort and retrieve the top_k file-similarity pairs
    file_similarity_pairs = [file_vector_pairs[i] for i in indices[0]]
    end = time.time()
    print(f"Time taken to find closest vectors faiss: {end - start} seconds")
    return file_similarity_pairs