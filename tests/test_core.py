from types import SimpleNamespace
import numpy as np
from rag.core import (
    Retriever, NO_ANSWER, answer_question, build_prompt, chunk_text, embed_texts, read_document,
)

class FakeEmbeddings:
    def create(self, model, input):
        def vector(t):
            t = t.lower()
            return [float('leave' in t or '休假' in t), float('password' in t or '密碼' in t), 0.1]
        data = [SimpleNamespace(index=i, embedding=vector(t)) for i, t in enumerate(input)]
        return SimpleNamespace(data=list(reversed(data)))

class FakeCompletions:
    def create(self, **kwargs):
        assert '文件' in kwargs['messages'][1]['content']
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='依據文件，可以申請休假。[S1]'))])

class FakeClient:
    embeddings = FakeEmbeddings()
    chat = SimpleNamespace(completions=FakeCompletions())


def test_chunk_overlap():
    chunks = chunk_text('abcdefghij', 'a.txt', size=6, overlap=2)
    assert [c.text for c in chunks] == ['abcdef', 'efghij']
    assert [c.chunk_id for c in chunks] == [1, 2]


def test_read_utf8():
    assert read_document('test.txt', '假期'.encode('utf-8')) == '假期'


def test_embeddings_normalization_and_order():
    matrix = embed_texts(FakeClient(), ['leave policy', 'password reset'])
    assert matrix.shape == (2, 3)
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1)
    assert matrix[0, 0] > matrix[0, 1]
    assert matrix[1, 1] > matrix[1, 0]


def test_retrieval_and_sources():
    chunks = [*chunk_text('Leave policy: 10 days', 'hr.txt'), *chunk_text('Password reset guide', 'it.txt')]
    hits = Retriever(chunks, FakeClient()).search('leave', top_k=2, threshold=0.5)
    assert len(hits) == 1
    assert hits[0][0].source == 'hr.txt'
    assert '[S1]' in build_prompt('leave?', hits)


def test_no_hits_no_api_call():
    assert answer_question('question', [], FakeClient()) == NO_ANSWER


def test_answer_with_citation():
    chunks = chunk_text('Leave policy', 'hr.txt')
    hits = Retriever(chunks, FakeClient()).search('leave')
    assert '[S1]' in answer_question('leave?', hits, FakeClient())
