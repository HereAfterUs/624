import jieba
from rank_bm25 import BM25Okapi
from zhipuai import ZhipuAI

client = ZhipuAI(api_key="22a6753b6eb44bb187a00a9bd6ef6a19.lbzGv63p3Glpgzqk")


class SimpleRAG:
    def __init__(self, file_path):
        # 读取知识库文档并按段落分割
        with open(file_path, 'r', encoding='utf-8') as f:
            self.paragraphs = [line.strip() for line in f if line.strip()]

        # 中文分词处理
        self.tokenized_paragraphs = [list(jieba.cut(p)) for p in self.paragraphs]
        self.bm25 = BM25Okapi(self.tokenized_paragraphs)

    def search(self, query, top_k=3):
        tokenized_query = list(jieba.cut(query))
        doc_scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_k]
        return [self.paragraphs[i] for i in top_indices]

    def generate(self, query):
        context = "\n".join(self.search(query))
        response = client.chat.completions.create(
            model="glm-4-plus",
            messages=[
                {
                    "role": "system",
                    "content": f"根据以下参考资料回答问题:\n{context}\n\n问题:{query}"
                },
                {
                    "role": "user",
                    "content": user_input
                }
            ],
            temperature=0.2,
            top_p=0.8
        )

        print(response.choices[0].message.content + "\n")


if __name__ == "__main__":
    rag = SimpleRAG("../files/knowledge.txt")
    user_input = "成都大学有多少国家级一流课程"
    rag.generate(user_input)
