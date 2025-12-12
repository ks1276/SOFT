# build_rag_index.py  (프로젝트 루트)
from rag.vectordb import RAGVectorStore

if __name__ == "__main__":
    pdf_dir = "./project_pdfs"       # 여기에 PDF 모아두기
    db_dir = "./chroma_db"

    vs = RAGVectorStore(db_dir=db_dir, collection_name="project_docs")
    n_chunks = vs.build_from_pdf_dir(
        pdf_dir=pdf_dir,
        chunk_size=800,
        chunk_overlap=200,
        reset=True,
    )
    print(f"색인 완료! 총 {n_chunks}개 청크가 저장되었습니다.")
