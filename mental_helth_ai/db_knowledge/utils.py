import nltk


def split_into_sentences(doc):
    """
    Splits the document content into sentences using NLTK's sentence tokenizer.

    Args:
    doc (Document): The document to be split.

    Returns:
    List[str]: List of sentences.
    """
    return nltk.sent_tokenize(doc.page_content)


def reconstruct_documents(sentences, target_lines_per_chunk=5):
    """
    Reconstructs documents from sentences, ensuring that chunks have
    approximately the target number of lines.

    Args:
    sentences (List[str]): List of sentences to be grouped into chunks.
    target_lines_per_chunk (int): Target number of lines per chunk.

    Returns:
    List[str]: List of document chunks.
    """
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split('\n'))
        if current_length + sentence_length > target_lines_per_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(sentence)
        current_length += sentence_length

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks
