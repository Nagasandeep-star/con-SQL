from src.llm.ollama_client import OllamaClient

client = OllamaClient()

sql = client.generate_sql(
    "Write SQL to list all employees from the employees table."
)

print(sql)