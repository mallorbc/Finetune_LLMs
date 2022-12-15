import mii
from transformers import AutoTokenizer
generator = mii.mii_query_handle("deployment")
result = generator.query({"query": ["DeepSpeed is"]}, do_sample=True, max_length=2048,top_k=50, top_p=0.95, temperature=1.0)
print(result)
