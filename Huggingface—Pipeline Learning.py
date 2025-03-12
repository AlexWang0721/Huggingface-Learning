# Hugging Face --Pipeline的API调用
from transformers import pipeline
import os

# 设置代理
proxy_url = 'http://127.0.0.1:7890'  # 替换为你的代理地址和端口
os.environ['HTTP_PROXY'] = proxy_url
os.environ['HTTPS_PROXY'] = proxy_url

# 1.执行零样本分类
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli",revision='d7645e1')
result = classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],)
print(result)

# 2.情感分析 sentiment-analysis
classifier = pipeline("sentiment-analysis",
                      model='distilbert/distilbert-base-uncased-finetuned-sst-2-english',
                      revision='714eb0f')
result2= classifier("I've been waiting for a HuggingFace course my whole life.")
print(result2)

# 3.文本生成 text-generation
generator = pipeline("text-generation",)
result3 = generator("In this course, we will teach you how to",
          num_return_sequences=1,
          max_length=20)
print(result3)

# 指定特定的模型
# DeepSeek-R1模型
generator = pipeline("text-generation", model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
result3_1= generator(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=2,
)
print(result3_1)

# Google/gemma-2模型
messages = [
    {"role": "user", "content": "In this course, we will teach you how to"},
]
pipe = pipeline("text-generation", model="google/gemma-2-2b-it")
output = pipe(messages,max_length=30,
              num_return_sequences=2)
print(output)

# 4. 填空 fill-mask
unmasker = pipeline("fill-mask")
result4= unmasker("This course will teach you all about <mask> models.", top_k=2)
print(result4)

# 5.命名实体识别 NER
ner = pipeline("ner", grouped_entities=True)
result5= ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")
print(result5)

# 6.问答 question-answering
question_answer = pipeline('question-answering')
result6= question_answer(
    question="Where do I work?",
    context="My name is Sylvain and I work at Hugging Face in Brooklyn",
)
print(result6)

# 7.总结 summarization
summarizer = pipeline("summarization")
result7= summarizer(
    """
    America has changed dramatically during recent years. Not only has the number of 
    graduates in traditional engineering disciplines such as mechanical, civil, 
    electrical, chemical, and aeronautical engineering declined, but in most of 
    the premier American universities engineering curricula now concentrate on 
    and encourage largely the study of engineering science. As a result, there 
    are declining offerings in engineering subjects dealing with infrastructure, 
    the environment, and related issues, and greater concentration on high 
    technology subjects, largely supporting increasingly complex scientific 
    developments. While the latter is important, it should not be at the expense 
    of more traditional engineering.

    Rapidly developing economies such as China and India, as well as other 
    industrial countries in Europe and Asia, continue to encourage and advance 
    the teaching of engineering. Both China and India, respectively, graduate 
    six and eight times as many traditional engineers as does the United States. 
    Other industrial countries at minimum maintain their output, while America 
    suffers an increasingly serious decline in the number of engineering graduates 
    and a lack of well-educated engineers.
"""
)
print(result7)
